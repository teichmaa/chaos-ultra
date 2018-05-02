package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.*;


import java.io.Closeable;
import java.nio.Buffer;
import java.text.SimpleDateFormat;
import java.util.Date;

import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static jcuda.driver.CUgraphicsRegisterFlags.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD;
import static jcuda.driver.CUresourcetype.CU_RESOURCE_TYPE_ARRAY;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

public class CudaLauncher implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;

    private AbstractFractalRenderKernel kernel;

    private CUdeviceptr deviceOut;
    /**
     * Actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
     */

    private int blockDimX = 32;
    private int blockDimY = 32;

    private CUgraphicsResource viewCudaResource = new CUgraphicsResource();
    private CUstream defaultStream = new CUstream();

    private cudaGraphicsResource r_resource = new cudaGraphicsResource();
    private cudaStream_t r_defaultStream = new cudaStream_t();

    public CudaLauncher(AbstractFractalRenderKernel kernel, int outputTextureGLhandle) {
        this.kernel = kernel;

        cudaInit();

        //copy the color palette to device:
//        int[] palette = createColorPalette();
//        CUdeviceptr devicePalette = new CUdeviceptr();
//        cuMemAlloc(devicePalette, palette.length * Sizeof.INT);
//        cuMemcpyHtoD(devicePalette, Pointer.to(palette),palette.length * Sizeof.INT);

        registerOutputTexture(outputTextureGLhandle);
    }

    public void registerOutputTexture(int textureGLhandle){
        //documentation: http://www.jcuda.org/jcuda/doc/jcuda/runtime/JCuda.html#cudaGraphicsGLRegisterImage(jcuda.runtime.cudaGraphicsResource,%20int,%20int,%20int)
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(r_resource, textureGLhandle, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    }

    public void unregisterOutputTexture(){
        jcuda.runtime.JCuda.cudaGraphicsUnregisterResource(r_resource);
    }

    /**
     * Allocate 2D array on device
     * Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
     * @param width
     * @param height
     * @return pitch
     */
    private long allocateOutputBuffer(int width, int height){
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

    public void launchKernel_RuntimeAPI(boolean verbose, Buffer outputBuffer) {
        long start = System.currentTimeMillis();

        int width = kernel.getWidth();
        int height = kernel.getHeight();

        jcuda.runtime.JCuda.cudaGraphicsMapResources(1, new cudaGraphicsResource[]{r_resource}, r_defaultStream);
        {
            jcuda.runtime.cudaArray arr = new cudaArray();
            jcuda.runtime.JCuda.cudaGraphicsSubResourceGetMappedArray(arr, r_resource, 0, 0);
            jcuda.runtime.cudaResourceDesc desc = new cudaResourceDesc();
            {
                desc.resType = cudaResourceTypeArray;
                desc.array_array = arr;
            }
            jcuda.runtime.cudaSurfaceObject surface = new cudaSurfaceObject();
            jcuda.runtime.JCuda.cudaCreateSurfaceObject(surface, desc);
            {
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
                {
                    kernelParamsArr[kernel.PARAM_IDX_SURFACE] = Pointer.to(surface);
                    kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(new int[]{0}); //pitch is obsollete
                    kernelParamsArr[kernel.PARAM_IDX_DEVICE_OUT] = Pointer.to(deviceOut);
                }
                Pointer kernelParams = Pointer.to(kernelParamsArr);

                CUfunction kernelFunction = kernel.getMainFunction();

                // The main thing - launching the kernel!:
                {
                    cuLaunchKernel(kernelFunction,
                            width / blockDimX, height / blockDimY, 1,
                            blockDimX, blockDimY, 1,
                            0, null,           // Shared memory size and defaultStream
                            kernelParams, null // Kernel- and extra parameters
                    );
                    cuCtxSynchronize();
                }
            }
            jcuda.runtime.JCuda.cudaDestroySurfaceObject(surface);
        }
        jcuda.runtime.JCuda.cudaGraphicsUnmapResources(1, new cudaGraphicsResource[]{r_resource}, r_defaultStream);

        long end = System.currentTimeMillis();
        if (verbose) {
            System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
        }

        if(outputBuffer == null) return;
        //copy to host:
        CUDA_MEMCPY2D copyInfo = new CUDA_MEMCPY2D();
        {
            copyInfo.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
            copyInfo.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
            copyInfo.srcDevice = deviceOut;
            copyInfo.dstHost = Pointer.to(outputBuffer);
            copyInfo.srcPitch = 0; //TODO pitch missing
            copyInfo.Height = height;
            copyInfo.WidthInBytes = width * Sizeof.INT;
        }
        cuMemcpy2D(copyInfo);
    }

    public void launchKernel(boolean verbose) {

        long start = System.currentTimeMillis();

        int width = kernel.getWidth();
        int height = kernel.getHeight();

        if (width % blockDimX != 0) {
            throw new CudaRendererException("Unsupported input parameter: width must be multiple of " + blockDimX);
        }
        if (height % blockDimY != 0) {
            throw new CudaRendererException("Unsupported input parameter: height must be multiple of " + blockDimY);
        }
        if (width > CUDA_MAX_GRID_DIM) {
            throw new CudaRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM);
        }
        if (height > CUDA_MAX_GRID_DIM) {
            throw new CudaRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM);
        }


        cuGraphicsMapResources(1, new CUgraphicsResource[]{viewCudaResource}, defaultStream);
        {
            CUarray array = new CUarray();
            cuGraphicsSubResourceGetMappedArray(array, viewCudaResource, 0, 0);

            CUDA_RESOURCE_DESC resource_desc = new CUDA_RESOURCE_DESC();
            {
                resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
                resource_desc.array_hArray = array;
            }

            CUsurfObject surface = new CUsurfObject();
            cuSurfObjectCreate(surface, resource_desc);
            //TODO, ok, tady to padá
            {
                // Set up the kernel parameters: A pointer to an array
                // of pointers which point to the actual values.
                NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
                {
                    kernelParamsArr[kernel.PARAM_IDX_SURFACE] = Pointer.to(surface);
                    kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(new int[]{0}); //pitch is obsollete
                    //todo druha funkce tady ma jeste jeden param
                }
                Pointer kernelParams = Pointer.to(kernelParamsArr);

                CUfunction kernelFunction = kernel.getMainFunction();

                // The main thing - launching the kernel!:
                long cudaKernelStartTime = System.currentTimeMillis();
                {
                    cuLaunchKernel(kernelFunction,
                            width / blockDimX, height / blockDimY, 1,
                            blockDimX, blockDimY, 1,
                            0, null,           // Shared memory size and defaultStream
                            kernelParams, null // Kernel- and extra parameters
                    );
                    cuCtxSynchronize();
                }
                long cudaKernelEndTime = System.currentTimeMillis();
                if (verbose)
                    System.out.println(kernel.getMainFunctionName() + " computed on CUDA in " + (cudaKernelEndTime - cudaKernelStartTime) + " ms");
                long createImageStartTime = cudaKernelEndTime;


            }
            cuSurfObjectDestroy(surface);

        }
        cuGraphicsUnmapResources(1, new CUgraphicsResource[]{viewCudaResource}, defaultStream);
        cuStreamSynchronize(defaultStream);

        long end = System.currentTimeMillis();
        if (verbose) {
            System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
        }


        /*
        {
            //copy to host:
            CUDA_MEMCPY2D copyInfo = new CUDA_MEMCPY2D();
            {
                copyInfo.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
                copyInfo.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
                copyInfo.srcDevice = deviceOut;
                copyInfo.dstHost = Pointer.to(outputBuffer);
                copyInfo.srcPitch = pitch;
                copyInfo.Height = height;
                copyInfo.WidthInBytes = width * Sizeof.INT;
            }
            cuMemcpy2D(copyInfo);
            //Note: we don't have to check memcpy return value, jcuda checks it for us (and will eventually throw an exception)
            long copyEndTime = System.currentTimeMillis();
        }*/
//        int i = 0;
//        boolean allnull = true;
//        while (outputBuffer.hasRemaining()){
//            if(((ByteBuffer) outputBuffer).getInt() != 0){
//                allnull = false;
//                break;
//               // System.out.println("i = " + i);
//            }
//            i++;
//        }
//        if(allnull && verbose)
//            System.out.println("All null :(");

//        if (verbose)
//            System.out.println("device to host copy finished in " + (copyEndTime - createImageStartTime) + " ms");


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
        cuMemFree(deviceOut);
        //      cuMemFree(devicePalette);
    }
}
