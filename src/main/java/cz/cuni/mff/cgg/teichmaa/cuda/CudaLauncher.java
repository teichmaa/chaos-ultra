package cz.cuni.mff.cgg.teichmaa.cuda;

import cz.cuni.mff.cgg.teichmaa.view.ImageHelpers;
import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.io.Closeable;
import java.nio.Buffer;
import java.security.InvalidParameterException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

public class CudaLauncher implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;

    private AbstractFractalRenderKernel kernel;

    private int blockDimX = 32;
    private int blockDimY = 32;

    private cudaGraphicsResource outputTextureResource = new cudaGraphicsResource();
    private cudaGraphicsResource paletteTextureResource = new cudaGraphicsResource();
    private cudaStream_t defaultStream = new cudaStream_t();

    public CudaLauncher(AbstractFractalRenderKernel kernel, int outputTextureGLhandle, int paletteTextureGLhandle) {
        this.kernel = kernel;

        cudaInit();

        //copy the color palette to device:
//        int[] palette = createColorPalette();
//        CUdeviceptr devicePalette = new CUdeviceptr();
//        cuMemAlloc(devicePalette, palette.length * Sizeof.INT);
//        cuMemcpyHtoD(devicePalette, Pointer.to(palette),palette.length * Sizeof.INT);

        registerOutputTexture(outputTextureGLhandle);
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(paletteTextureResource, paletteTextureGLhandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
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
     * @param deviceOut output parameter, will contain pointer to allocated memory
     * @return pitch
     */
    private long allocateDeviceOutputBuffer(int width, int height, CUdeviceptr deviceOut) {

        /**
         * Pitch = actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
         */
        long pitch;
        long[] pitchptr = new long[1];

        deviceOut = new CUdeviceptr();
        cuMemAllocPitch(deviceOut, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);

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

    private void copyFromDevToHost(Buffer hostOut, int width, int height, long pitch, CUdeviceptr deviceOut) {
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
            cudaSurfaceObject surfaceOut = new cudaSurfaceObject();
            {
                cudaResourceDesc surfaceDesc = new cudaResourceDesc();
                {
                    cudaArray arr = new cudaArray();
                    JCuda.cudaGraphicsSubResourceGetMappedArray(arr, outputTextureResource, 0, 0);
                    surfaceDesc.resType = cudaResourceTypeArray;
                    surfaceDesc.array_array = arr;
                }
                JCuda.cudaCreateSurfaceObject(surfaceOut, surfaceDesc);
            }
            cudaTextureObject texturePalette = new cudaTextureObject();
            {
                cudaResourceDesc resourceDesc = new cudaResourceDesc();
                {
                    cudaArray arr = new cudaArray();
                    JCuda.cudaGraphicsSubResourceGetMappedArray(arr, paletteTextureResource, 0, 0);
                    resourceDesc.resType = cudaResourceTypeArray;
                    resourceDesc.array_array = arr;
                }
                cudaTextureDesc textureDesc = new cudaTextureDesc();
                {
                    //nothing, use default (0) values
                }
                JCuda.cudaCreateTextureObject(texturePalette, resourceDesc, textureDesc, null);
            }
            try {
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
                {
                    kernelParamsArr[kernel.PARAM_IDX_SURFACE_OUT] = Pointer.to(surfaceOut);
                    kernelParamsArr[kernel.PARAM_IDX_TEX_PALETTE] = Pointer.to(texturePalette);
                    kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(new int[]{0}); //pitch is obsollete, no longer used by my kernels
                    kernelParamsArr[kernel.PARAM_IDX_DEVICE_OUT] = Pointer.to(new int[]{0}); //device out is obsollete, no longer used by my kernels
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
                JCuda.cudaDestroySurfaceObject(surfaceOut);
               // JCuda.cudaDestroySurfaceObject(surfacePalette);
                //JCuda.cudaDestroyTextureObject(texturePalette);
            }
        } finally {
            JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{outputTextureResource}, defaultStream);
            JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{paletteTextureResource}, defaultStream);
        }

        long end = System.currentTimeMillis();
        if (verbose) {
            System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
        }

        //coloring:
//        int[] p = createColorPalette();
//        for (int i = 0; i < hostOut.length; i++) {
//            //int pIdx = p.length - (Math.round(hostOut[i] / (float) dwell * p.length));
//            int pIdx = p.length - hostOut[i] % p.length;
//            pIdx = Math.max(0, Math.min(p.length - 1, pIdx));
//            final int FULL_OPACITY_MASK = 0xff000000;
//            hostOut[i] = p[pIdx] | FULL_OPACITY_MASK;
//
//        }

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
    }
}
