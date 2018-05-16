package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.io.Closeable;
import java.nio.Buffer;
import java.security.InvalidParameterException;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

public class FractalRenderer implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;
    public static final int SUPER_SAMPLING_MAX_LEVEL = 256;
    private static final Pointer NULLPTR = Pointer.to(new byte[0]);

    static {
        CudaHelpers.cudaInit();
    }

    private RenderingKernel kernel;
    private FractalRenderingModule module;

    private int blockDimX = 32;
    private int blockDimY = 32;
    private int paletteLength;

    //private CUdeviceptr devout_debug = new CUdeviceptr();
    //private long devout_pitch_debug;
    private CUdeviceptr randomValues;

    private cudaGraphicsResource outputTextureResource = new cudaGraphicsResource();
    private cudaGraphicsResource paletteTextureResource = new cudaGraphicsResource();
    private cudaStream_t defaultStream = new cudaStream_t();

    public FractalRenderer(FractalRenderingModule module, int outputTextureGLHandle, int outputTextureGLTarget, int paletteTextureGLHandle, int paletteTextureGLTarget, int paletteLength) {
        this.module = module;
        kernel = module.getKernel(KernelUnderSampled.class);
        this.paletteLength = paletteLength;

        // devout_pitch_debug = allocateDevice2DBuffer(kernel.getWidth(), kernel.getHeight(), devout_debug);

        registerOutputTexture(outputTextureGLHandle, outputTextureGLTarget);
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(paletteTextureResource, paletteTextureGLHandle, paletteTextureGLTarget, cudaGraphicsRegisterFlagsReadOnly);

        //moduleInit();

        randomSamplesInit();
    }

    private void randomSamplesInit() {
//        if(randomValues != null){
//            JCuda.cudaFree(randomValues);
//        }else{
//            randomValues = new CUdeviceptr();
//        }
//
//        int w = kernel.getWidth();
//        int h = kernel.getHeight();
//
//        int sampleCount = w * h * SUPER_SAMPLING_MAX_LEVEL * 2; //2 because we are in 2D
//
//        curandGenerator gen = new curandGenerator();
//        //JCurand.curandCreateGenerator(gen, CURAND_RNG_QUASI_SOBOL32);
//        JCurand.curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
//        //JCurand.curandSetQuasiRandomGeneratorDimensions(gen, 2);
//        JCuda.cudaMalloc(randomValues, sampleCount * Sizeof.FLOAT);
///*
//        //crazy slow and hacky solution, which yet might give desired results
//        for (int i = 0; i < w; i++) {
//            for (int j = 0; j < h; j++) {
//                PointerHelpers.nativePointerArtihmeticHack(randomValues, ssl * Sizeof.FLOAT);
//                JCurand.curandGenerateUniform(gen, randomValues, ssl);
//            }
//        }
//        PointerHelpers.nativePointerArtihmeticHack(randomValues, - w * h * ssl * Sizeof.FLOAT);*/
//        JCurand.curandGenerateUniform(gen, randomValues, sampleCount);
//
//        /*ByteBuffer testOut = ByteBuffer.allocateDirect(sampleCount * Sizeof.FLOAT);
//        JCuda.cudaMemcpy(Pointer.to(testOut), randomValues, sampleCount * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//        float[] floats = new float[sampleCount];
//        for (int i = 0; i < floats.length; i++) {
//            floats[i] = (float) testOut.getInt();
//        }
//        for (int i = 0; i < 100; i++) {
//            System.out.println(floats[i]);
//        }*/
//        int a = 0; //breakpoint
    }


    private void moduleInit() {

        KernelInit kernel = module.getKernel(KernelInit.class);

        Pointer kernelParams = Pointer.to(kernel.getKernelParams());
        cuLaunchKernel(kernel.getFunction(),
                1, 1, 1,
                1, 1, 1,
                0, null,
                kernelParams, null
        );
        cuCtxSynchronize();
    }

    public void registerOutputTexture(int outputTextureGLhandle, int GLtarget) {
        //documentation: http://www.jcuda.org/jcuda/doc/jcuda/runtime/JCuda.html#cudaGraphicsGLRegisterImage(jcuda.runtime.cudaGraphicsResource,%20int,%20int,%20int)
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(outputTextureResource, outputTextureGLhandle, GLtarget, cudaGraphicsRegisterFlagsWriteDiscard);
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

    public void resize(int width, int height, int outputTextureGLhandle, int GLtarget) {
        //System.out.println("resize: " +width + " x " + height);
        kernel.setWidth(width);
        kernel.setHeight(height);
        randomSamplesInit();
        registerOutputTexture(outputTextureGLhandle, GLtarget);
    }

    public int getWidth() {
        return kernel.getWidth();
    }

    public int getHeight() {
        return kernel.getHeight();
    }

    /**
     * @param async when true, the function will return just after launching the kernel and will not wait for it to end. The bitmaps will still be synchronized.
     */
    public void launchKernel(boolean async) {
        launchKernel(async, kernel);
    }

    private void launchKernel(boolean async, RenderingKernel kernel) {
        long start = System.currentTimeMillis();

        int width = kernel.getWidth();
        int height = kernel.getHeight();

        try {
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
                        kernelParamsArr[kernel.PARAM_IDX_PALETTE_LENGTH] = Pointer.to(new int[]{paletteLength});
                        if(kernel instanceof KernelMain)
                            kernelParamsArr[((KernelMain)kernel).PARAM_IDX_RANDOM_SAMPLES] = Pointer.to(randomValues);
                    }
                    Pointer kernelParams = Pointer.to(kernelParamsArr);
                    CUfunction kernelFunction = kernel.getFunction();
                    int gridDimX = width / blockDimX;
                    int gridDimY = height / blockDimY;

                    if (gridDimX <= 0 || gridDimY <= 0) return;
                    if (gridDimX > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM * blockDimX);
                    }
                    if (gridDimY > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM * blockDimY);
                    }
                    try {
                        cuLaunchKernel(kernelFunction,
                                gridDimX, gridDimY, 1,
                                blockDimX, blockDimY, 1,
                                0, null,           // Shared memory size and defaultStream
                                kernelParams, null // Kernel- and extra parameters
                        );
                        if (!async)
                            cuCtxSynchronize();
                    } catch (CudaException e) {
                        System.err.println("Error just after launching a kernel:");
                        System.err.println(e);
                    }
                } finally {
                    JCuda.cudaDestroySurfaceObject(surfaceOutput);
                    JCuda.cudaDestroySurfaceObject(surfacePalette);
                }
            } finally {
                JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{outputTextureResource}, defaultStream);
                JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{paletteTextureResource}, defaultStream);
            }
        } catch (CudaException | FractalRendererException e) {
            System.err.println("Error during kernel launch preparation:");
            System.err.println(e);
        }

        long end = System.currentTimeMillis();
        //System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
    }

    @Override
    public void close() {
        unregisterOutputTexture();
        //cuMemFree(deviceOut);
        //      cuMemFree(devicePalette);
        if (randomValues != null)
            JCuda.cudaFree(randomValues);
    }

    public void setAdaptiveSS(boolean adaptiveSS) {
        if(!(this.kernel instanceof KernelMain))
            return;
        KernelMain kernelMain = (KernelMain) this.kernel;
        kernelMain.setAdaptiveSS(adaptiveSS);
    }

    public void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        if(!(this.kernel instanceof KernelMain))
            return;
        KernelMain kernelMain = (KernelMain) this.kernel;
        kernelMain.setVisualiseAdaptiveSS(visualiseAdaptiveSS);
    }

    public void setSuperSamplingLevel(int supSampLvl) {
        if(!(this.kernel instanceof KernelMain))
            return;
        KernelMain kernelMain = (KernelMain) this.kernel;
        kernelMain.setSuperSamplingLevel(supSampLvl);
        //randomSamplesInit();
    }

    public void setDwell(int dwell) {
        kernel.setDwell(dwell);
    }

    public int getSuperSamplingLevel() {
        if(!(this.kernel instanceof KernelMain))
            return -1;
        KernelMain kernelMain = (KernelMain) this.kernel;
        return kernelMain.getSuperSamplingLevel();
    }

    public int getDwell() {
        return kernel.getDwell();
    }

    public void setBounds(float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        kernel.setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

}
