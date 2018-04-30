package cz.cuni.mff.cgg.teichmaa;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import java.io.Closeable;
import java.nio.Buffer;
import java.text.SimpleDateFormat;
import java.util.Date;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;

public class CudaLauncher implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;

    private AbstractFractalRenderKernel kernel;
    private int width;
    private int height;
    private CUdeviceptr deviceOut;
    /**
     * Actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
     */
    private long pitch;
    private long[] pitchptr = new long[1];
    private int blockDimX = 32;
    private int blockDimY = 32;

    public CudaLauncher(AbstractFractalRenderKernel kernel) {
        this.kernel = kernel;
        this.width = kernel.getWidth();
        this.height = kernel.getHeight();

        cudaInit();

        // Allocate 2D array on device
        // Note: Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
        deviceOut = new CUdeviceptr();
        cuMemAllocPitch(deviceOut, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);
        //Note: we don't have to check malloc return value, jcuda checks it for us (and will eventually throw an exception)
        pitch = pitchptr[0];
        if (pitch <= 0) {
            throw new CudaException("cuMemAllocPitch returned pitch with value 0 (or less)");
        }

        if (pitch > Integer.MAX_VALUE) {
            //this would mean an array with length bigger that Integer.MAX_VALUE and this is not supported by Java
            throw new CudaException("Pitch is too big (bigger than Integer.MAX_VALUE): " + pitch);
        }

        //copy the color palette to device:
//        int[] palette = createColorPalette();
//        CUdeviceptr devicePalette = new CUdeviceptr();
//        cuMemAlloc(devicePalette, palette.length * Sizeof.INT);
//        cuMemcpyHtoD(devicePalette, Pointer.to(palette),palette.length * Sizeof.INT);
    }

    private CUdevice cudaInit() {

        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);

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

    public void launchKernel(Buffer outputBuffer, boolean verbose) {

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

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
        kernelParamsArr[kernel.PARAM_IDX_DEVICE_OUT] = Pointer.to(deviceOut);
        kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(pitchptr);
        Pointer kernelParams = Pointer.to(kernelParamsArr);

        CUfunction kernelFunction = kernel.getMainFunction();

        // The main thing - launching the kernel!:
        long cudaKernelStartTime = System.currentTimeMillis();
        cuLaunchKernel(kernelFunction,
                width / blockDimX, height / blockDimY, 1,
                blockDimX, blockDimY, 1,
                0, null,           // Shared memory size and stream
                kernelParams, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        long cudaKernelEndTime = System.currentTimeMillis();
        if(verbose)
            System.out.println(kernel.getMainFunctionName() + " computed on CUDA in " + (cudaKernelEndTime - cudaKernelStartTime) + " ms");

        long createImageStartTime = cudaKernelEndTime;

        int deviceActualWidth = (int) (pitch / Sizeof.INT);
        if ((long) deviceActualWidth * (long) height > (long) Integer.MAX_VALUE) {
            throw new CudaException("pitch*height/Sizeof.INT is too big (bigger than Integer.MAX_VALUE) and cannot be handled by Java: " + pitch);
        }

        //copy to host:
        CUDA_MEMCPY2D copyInfo = new CUDA_MEMCPY2D();
        copyInfo.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        copyInfo.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copyInfo.srcDevice = deviceOut;
        copyInfo.dstHost = Pointer.to(outputBuffer);
        copyInfo.srcPitch = pitch;
        copyInfo.Height = height;
        copyInfo.WidthInBytes = width * Sizeof.INT;
        cuMemcpy2D(copyInfo);
        //Note: we don't have to check memcpy return value, jcuda checks it for us (and will eventually throw an exception)
        long copyEndTime = System.currentTimeMillis();

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

        if(verbose)
            System.out.println("device to host copy finished in " + (copyEndTime - createImageStartTime) + " ms");


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


    private void saveAsImage(long createImageStartTime, int renderTimeTotalMs, int[] data){
        int dwell = kernel.getDwell();

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
