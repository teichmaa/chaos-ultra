package cz.cuni.mff.cgg.teichmaa.cuda;

import cz.cuni.mff.cgg.teichmaa.view.ImageHelpers;
import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.*;

import java.io.Closeable;
import java.nio.Buffer;
import java.security.InvalidParameterException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

public class CudaLauncher implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;
    private static final Pointer NULLPTR = Pointer.to(new byte[0]);

    private AbstractFractalRenderKernel kernel;

    private int blockDimX = 32;
    private int blockDimY = 32;
    private int paletteLength;

    private CUdeviceptr devout_debug = new CUdeviceptr();
    private long devout_pitch_debug;
    private CUdeviceptr randomValues;

    private cudaGraphicsResource outputTextureResource = new cudaGraphicsResource();
    private cudaGraphicsResource paletteTextureResource = new cudaGraphicsResource();
    private cudaStream_t defaultStream = new cudaStream_t();

    public CudaLauncher(AbstractFractalRenderKernel kernel, int outputTextureGLhandle, int paletteTextureGLhandle, int paletteLength) {
        this.kernel = kernel;
        this.paletteLength = paletteLength;

        cudaInit();
        // devout_pitch_debug = allocateDevice2DBuffer(kernel.getWidth(), kernel.getHeight(), devout_debug);

        registerOutputTexture(outputTextureGLhandle);
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(paletteTextureResource, paletteTextureGLhandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);

        //kernelInit();

        kernel.setSuperSamplingLevel(128);
        randomSamplesInit();
        kernel.setSuperSamplingLevel(16);
        kernel.setDwell(500);

    }

    private void randomSamplesInit(){
        //todo tohle nefunguje pri resize
        //  > proste CudaLauncheru pridej bud metodu resize nebo observer na kernel width-has-changed (setwh spoj do setDimensions > jediny callback)
        int w = kernel.getWidth();
        int h = kernel.getHeight();
        int ssl = kernel.getSuperSamplingLevel();

        int sampleCount = w* h *ssl;

        curandGenerator gen = new curandGenerator();
        //JCurand.curandCreateGenerator(gen, CURAND_RNG_QUASI_SOBOL32);
        JCurand.curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
        //JCurand.curandSetQuasiRandomGeneratorDimensions(gen, 2);
        randomValues = new CUdeviceptr();
        JCuda.cudaMalloc(randomValues, sampleCount * Sizeof.FLOAT);
        /*
        //crazy slow and hacky solution, which yet might give desired results
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                PointerHelpers.nativePointerArtihmeticHack(randomValues, ssl * Sizeof.FLOAT);
                JCurand.curandGenerateUniform(gen, randomValues, ssl);
            }
        }
        PointerHelpers.nativePointerArtihmeticHack(randomValues, - w * h * ssl * Sizeof.FLOAT);*/
        JCurand.curandGenerateUniform(gen, randomValues, sampleCount);


        /*ByteBuffer testOut = ByteBuffer.allocateDirect(sampleCount * Sizeof.FLOAT);
        JCuda.cudaMemcpy(Pointer.to(testOut), randomValues, sampleCount * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
        float[] floats = new float[sampleCount];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = (float) testOut.getInt();
        }
        for (int i = 0; i < 100; i++) {
            System.out.println(floats[i]);
        }*/
        int a = 0;
    }


    private void kernelInit() {
        if (!kernel.isInitiable())
            return;

        Pointer kernelParams = Pointer.to(NULLPTR);
        cuLaunchKernel(kernel.getInitFunction(),
                1, 1, 1,
                1, 1, 1,
                0, null,
                kernelParams, null
        );
        cuCtxSynchronize();

    }

    public void registerOutputTexture(int outputTextureGLhandle) {
        //documentation: http://www.jcuda.org/jcuda/doc/jcuda/runtime/JCuda.html#cudaGraphicsGLRegisterImage(jcuda.runtime.cudaGraphicsResource,%20int,%20int,%20int)
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(outputTextureResource, outputTextureGLhandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    }

    public void unregisterOutputTexture() {
        jcuda.runtime.JCuda.cudaGraphicsUnregisterResource(outputTextureResource);
    }

    /**
     * Allocate 2D array on device
     * Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
     *
     * @param width
     * @param height
     * @param target output parameter, will contain pointer to allocated memory
     * @return pitch
     */
    private long allocateDevice2DBuffer(int width, int height, CUdeviceptr target) {

        /**
         * Pitch = actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
         */
        long pitch;
        long[] pitchptr = new long[1];

        cuMemAllocPitch(target, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);

        pitch = pitchptr[0];
        if (pitch <= 0) {
            throw new CudaException("cuMemAllocPitch returned pitch with value 0 (or less)");
        }

        if (pitch > Integer.MAX_VALUE) {
            //this would mean an array with length bigger that Integer.MAX_VALUE and this is not supported by Java
            throw new CudaException("Pitch is too big (bigger than Integer.MAX_VALUE): " + pitch);
        }
        return pitch;
    }

    private void copy2DFromDevToHost(Buffer hostOut, int width, int height, long pitch, CUdeviceptr deviceOut) {
        if (hostOut.capacity() < width * height * 4)
            throw new InvalidParameterException("Output buffer must be at least width * height * 4 bytes long. Buffer capacity: " + hostOut.capacity());
        //copy kernel result to host memory:
        CUDA_MEMCPY2D copyInfo = new CUDA_MEMCPY2D();
        {
            copyInfo.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
            copyInfo.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
            copyInfo.srcDevice = deviceOut;
            copyInfo.dstHost = Pointer.to(hostOut);
            copyInfo.srcPitch = pitch;
            copyInfo.Height = height;
            copyInfo.WidthInBytes = width * Sizeof.INT;
        }
        cuMemcpy2D(copyInfo);
    }

    private CUdevice cudaInit() {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);

        CUdevice dev = new CUdevice();
        CUcontext ctx = new CUcontext();
        cuDeviceGet(dev, 0);
        cuCtxCreate(ctx, 0, dev);

        return dev;
    }

    public int getBlockDimX() {
        return blockDimX;
    }

    public void setBlockDimX(int blockDimX) {
        this.blockDimX = blockDimX;
    }

    public int getBlockDimY() {
        return blockDimY;
    }

    public void setBlockDimY(int blockDimY) {
        this.blockDimY = blockDimY;
    }

    public AbstractFractalRenderKernel getKernel() {
        return kernel;
    }

    private boolean already = false;

    /**
     * @param verbose whether to print out how long the rendering has taken
     * @param async   when true, the function will return just after launching the kernel and will not wait for it to end. The bitmaps will still be synchronized.
     */
    public void launchKernel(boolean verbose, boolean async) {
        long start = System.currentTimeMillis();

        //if(already) return;
        //already = true;

        int width = kernel.getWidth();
        int height = kernel.getHeight();

        JCuda.cudaGraphicsMapResources(1, new cudaGraphicsResource[]{outputTextureResource}, defaultStream);
        JCuda.cudaGraphicsMapResources(1, new cudaGraphicsResource[]{paletteTextureResource}, defaultStream);
        try {
            cudaSurfaceObject surfaceOutput = new cudaSurfaceObject();
            {
                cudaResourceDesc surfaceDesc = new cudaResourceDesc();
                {
                    cudaArray arr = new cudaArray();
                    JCuda.cudaGraphicsSubResourceGetMappedArray(arr, outputTextureResource, 0, 0);
                    surfaceDesc.resType = cudaResourceTypeArray;
                    surfaceDesc.array_array = arr;
                }
                JCuda.cudaCreateSurfaceObject(surfaceOutput, surfaceDesc);
            }
            cudaSurfaceObject surfacePalette = new cudaSurfaceObject();
            {
                cudaResourceDesc surfaceDesc = new cudaResourceDesc();
                {
                    cudaArray arr = new cudaArray();
                    JCuda.cudaGraphicsSubResourceGetMappedArray(arr, paletteTextureResource, 0, 0);
                    surfaceDesc.resType = cudaResourceTypeArray;
                    surfaceDesc.array_array = arr;
                }
                JCuda.cudaCreateSurfaceObject(surfacePalette, surfaceDesc);
            }
            try {
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
                {
                    kernelParamsArr[kernel.PARAM_IDX_SURFACE_OUT] = Pointer.to(surfaceOutput);
                    kernelParamsArr[kernel.PARAM_IDX_SURFACE_PALETTE] = Pointer.to(surfacePalette);
                    kernelParamsArr[kernel.PARAM_IDX_RANDOM_SAMPLES] = Pointer.to(randomValues);
                    kernelParamsArr[kernel.PARAM_IDX_PALETTE_LENGTH] = Pointer.to(new int[]{paletteLength}); //device out is obsolete, only used for debugging
                    kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(new long[]{devout_pitch_debug}); //pitch is obsolete, only used for debugging
                    kernelParamsArr[kernel.PARAM_IDX_DEVICE_OUT] = Pointer.to(devout_debug); //device out is obsolete, only used for debugging
                }
                Pointer kernelParams = Pointer.to(kernelParamsArr);
                CUfunction kernelFunction = kernel.getMainFunction();
                int gridDimX = width / blockDimX;
                int gridDimY = height / blockDimY;

                if (gridDimX <= 0 || gridDimY <= 0) return;
                if (gridDimX > CUDA_MAX_GRID_DIM) {
                    throw new CudaRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM * blockDimX);
                }
                if (gridDimY > CUDA_MAX_GRID_DIM) {
                    throw new CudaRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM * blockDimY);
                }

                cuLaunchKernel(kernelFunction,
                        gridDimX, gridDimY, 1,
                        blockDimX, blockDimY, 1,
                        0, null,           // Shared memory size and defaultStream
                        kernelParams, null // Kernel- and extra parameters
                );
                if (!async)
                    cuCtxSynchronize();
            } finally {
                JCuda.cudaDestroySurfaceObject(surfaceOutput);
                JCuda.cudaDestroySurfaceObject(surfacePalette);
            }
        } finally {
            JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{outputTextureResource}, defaultStream);
            JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{paletteTextureResource}, defaultStream);
        }

        long end = System.currentTimeMillis();
        if (verbose) {
            System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
        }
    }


    private void saveAsImage(long createImageStartTime, int renderTimeTotalMs, int[] data) {
        int dwell = kernel.getDwell();
        int width = kernel.getWidth();
        int height = kernel.getHeight();

        String directoryPath = "E:\\Tonda\\Desktop\\fractal-out";
        //String fileName = directoryPath+"\\"+ new SimpleDateFormat("dd.MM.yy_HH-mm-ss").format(new Date()) +"_"+ kernel.getMainFunctionName()+ ".tiff";
        String juliaResult = "";
        if (kernel instanceof JuliaKernel) {
            JuliaKernel j = (JuliaKernel) kernel;
            juliaResult = "__c-" + j.getCx() + "+" + j.getCy() + "i";
        }
        String fileName = directoryPath + "\\" + new SimpleDateFormat("dd-MM-YY_mm-ss").format(new Date()) + "_" + kernel.getMainFunctionName().substring(0, 5)
                + "__dwell-" + dwell
                + "__res-" + width + "x" + height
                + "__time-" + renderTimeTotalMs + "ms"
                + juliaResult
                + ".tiff";
        ImageHelpers.createImage(data, width, height, fileName, "tiff");
        long createImageEndTime = System.currentTimeMillis();
        System.out.println(kernel.getMainFunctionName() + " saved to disk in " + (createImageEndTime - createImageStartTime) + " ms. Location: " + fileName);

    }

    @Override
    public void close() {
        //cuMemFree(deviceOut);
        //      cuMemFree(devicePalette);
        JCuda.cudaFree(randomValues);
    }
}
