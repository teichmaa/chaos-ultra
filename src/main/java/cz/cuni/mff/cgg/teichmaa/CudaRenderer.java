package cz.cuni.mff.cgg.teichmaa;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public class CudaRenderer {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;

    private static CUdevice cudaInit() {

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

    public static int[] launch(RenderingKernel kernel, boolean saveImage) {


        int width = kernel.getWidth();
        int height = kernel.getHeight();
        int blockDimX = 32;
        int blockDimY = 32;

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

        cudaInit();

        // Allocate 2D array on device
        // Note: Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
        CUdeviceptr deviceOut = new CUdeviceptr();
        /**
         * Actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
         */
        long pitch;
        long[] pitchptr = new long[1];
        cuMemAllocPitch(deviceOut, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);
        //Note: we don't have to check malloc return value, jcuda checks it for us (and will eventually throw an exception)
        pitch = pitchptr[0];
        if (pitch <= 0) {
            throw new CudaException("cuMemAllocPitch returned pitch with value 0 (or less)");
        }

        //copy the color palette to device:
//        int[] palette = createColorPalette();
//        CUdeviceptr devicePalette = new CUdeviceptr();
//        cuMemAlloc(devicePalette, palette.length * Sizeof.INT);
//        cuMemcpyHtoD(devicePalette, Pointer.to(palette),palette.length * Sizeof.INT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
        kernelParamsArr[kernel.PARAM_IDX_DEVICE_OUT] = Pointer.to(deviceOut);
        kernelParamsArr[kernel.PARAM_IDX_PITCH] = Pointer.to(pitchptr);
        Pointer kernelParams = Pointer.to(kernelParamsArr);

        //debug:
//        for (int i = 0; i < kernelParamsArr.length; i++) {
//            System.out.println("kernelParamsArr["+i+"] =\t" + kernelParamsArr[i]);
//        }
//
//        System.out.println("kernel = " + kernel);

//        kernelParamsArr[((MandelbrotKernel)kernel).PARAM_IDX_DWELL] = Pointer.to(new int[]{200000});
//        kernelParamsArr[((MandelbrotKernel)kernel).PARAM_IDX_WIDTH] = Pointer.to(new int[]{2048});
//        kernelParamsArr[((MandelbrotKernel)kernel).PARAM_IDX_HEIGHT] = Pointer.to(new int[]{2048});

        //debug only. Debugging kernelParams configuration:
//        Pointer kernelParams = Pointer.to(
//                Pointer.to(deviceOut),
//                Pointer.to(pitchptr),
////                Pointer.to(palette),
//                Pointer.to(new int[]{2048}),
//                Pointer.to(new int[]{2048}),
//                Pointer.to(new float[]{-1f}),
//                Pointer.to(new float[]{-1f}),
//                Pointer.to(new float[]{1f}),
//                Pointer.to(new float[]{1f}),
//                Pointer.to(new int[]{200})
//        );

        CUfunction kernelFunction = kernel.getMainFunction();
        // The main thing - launching the kernel!:
        long cudaKernelStartTime = System.currentTimeMillis();
        cuLaunchKernel(kernelFunction,
                width / blockDimX, height / blockDimY, 1,           // Grid dimension
                blockDimX, blockDimY, 1,  // Block dimension
                0, null,           // Shared memory size and stream
                kernelParams, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        long cudaKernelEndTime = System.currentTimeMillis();
        System.out.println(kernel.getMainFunctionName() + " computed on CUDA in " + (cudaKernelEndTime - cudaKernelStartTime) + " ms");

        long createImageStartTime = cudaKernelEndTime;
        //allocate host output
        if (pitch > Integer.MAX_VALUE) {
            throw new CudaException("Pitch is too big (bigger than Integer.MAX_VALUE) and cannot be handled by Java: " + pitch);
        }
        int deviceActualWidth = (int) (pitch / Sizeof.INT);
        if ((long) deviceActualWidth * (long) height > (long) Integer.MAX_VALUE) {
            throw new CudaException("pitch*height/Sizeof.INT is too big (bigger than Integer.MAX_VALUE) and cannot be handled by Java: " + pitch);
        }
        int hostOut[] = new int[deviceActualWidth * height];


        //copy to host:
        CUDA_MEMCPY2D copyInfo = new CUDA_MEMCPY2D();
        copyInfo.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        copyInfo.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copyInfo.srcDevice = deviceOut;
        copyInfo.dstHost = Pointer.to(hostOut);
        copyInfo.srcPitch = pitch;
        copyInfo.Height = height;
        copyInfo.WidthInBytes = width * Sizeof.INT;
        cuMemcpy2D(copyInfo);
        //Note: we don't have to check memcpy return value, jcuda checks it for us (and will eventually throw an exception)
        long copyEndTime = cudaKernelEndTime;

        System.out.println("device to host copy finished in " + (copyEndTime - createImageStartTime) + " ms");
        cuMemFree(deviceOut);
        //      cuMemFree(devicePalette);


        //debug:
//        printArray(arrayFlatTo2D(hostOut, width, height));
//        System.out.println("hostOut[0] = " + hostOut[0]);
//        System.out.println("hostOut[1] = " + hostOut[1]);
//        System.out.println("hostOut[2] = " + hostOut[2]);
//        System.out.println("hostOut[3] = " + hostOut[3]);
//        System.out.println("hostOut[4] = " + hostOut[4]);

        int dwell = -1;
        if (kernel instanceof MandelbrotKernel)
            dwell = ((MandelbrotKernel) kernel).getDwell();
        if (kernel instanceof JuliaKernel)
            dwell = ((JuliaKernel) kernel).getDwell();
        //todo wtf I need to have a common API for dwell


        //coloring:
        int[] p = createColorPalette();
        for (int i = 0; i < hostOut.length; i++) {
            //int pIdx = p.length - (Math.round(hostOut[i] / (float) dwell * p.length));
            int pIdx = p.length - hostOut[i] % p.length;
            pIdx = Math.max(0, Math.min(p.length - 1, pIdx));
            final int FULL_OPACITY_MASK = 0xff000000;
            hostOut[i] = p[pIdx] | FULL_OPACITY_MASK;

        }

        //See the result:
        if (saveImage) {

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
                    + "__time-" + (cudaKernelEndTime - cudaKernelStartTime) + "ms"
                    + juliaResult
                    + ".tiff";
            createImage(hostOut, deviceActualWidth, height, fileName);
            long createImageEndTime = System.currentTimeMillis();
            System.out.println(kernel.getMainFunctionName() + " saved to disk in " + (createImageEndTime - createImageStartTime) + " ms. Location: " + fileName);
        }

        return hostOut;

    }

    private static int[] createColorPalette() {
        int max = 256;
        int fullColor = 255;
        int[] p = new int[max * 6];

        //few rgb-2-hsv transitions palette:
        for (int i = 0; i < max; i++) {
            p[i] = fromRGB(i, 0, 0); //black to red gradient
        }

//        for (int i = max; i < max * 2; i++) {
//            p[i] = fromRGB(fullColor-i, fullColor-i, fullColor-i); //BW
//        }

        for (int i = max; i < max * 2; i++) {
            p[i] = fromRGB(fullColor, i, 0); //red to yellow gradient
        }

        for (int i = max * 2; i < max * 3; i++) {
            p[i] = fromRGB(fullColor-i,fullColor,0); //yellow to green gradient
        }

        for (int i = max * 3; i < max * 4; i++) {
            p[i] = fromRGB(0,fullColor,i); //green to cyan gradient
        }

        for (int i = max * 4; i < max * 5; i++) {
            p[i] = fromRGB(0,fullColor-i,fullColor); //cyan to blue gradient
        }

        for (int i = max * 5; i < max * 6; i++) {
            p[i] = fromRGB(0,0,fullColor-i); //blue to black gradient
        }

        return p;
    }

    private static int fromRGB(int r, int g, int b) {
        r <<= 16;
        g <<= 8;
        int rMask = 0xff0000;
        int gMask = 0x00ff00;
        int bMask = 0x0000ff;
        r &= rMask;
        g &= gMask;
        b &= bMask;
        return r | g | b;
    }

    private static int[][] arrayFlatTo2D(int[] a, int width, int height) {
        int result2D[][] = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                result2D[i][j] = a[j * width + i];
            }
        }
        return result2D;
    }

    private static int[] arrayFlatten(int[][] a) {
        int width = a.length;
        int height = a[0].length;
        int[] result = new int[width * height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                result[j * width + i] = a[i][j];
            }
        }
        return result;
    }

    private static void createImage(int[] rgbs, int width, int height, String fileName) {


        DataBuffer rgbData = new DataBufferInt(rgbs, rgbs.length);

        WritableRaster raster = Raster.createPackedRaster(rgbData, width, height, width,
                new int[]{0xff0000, 0xff00, 0xff},
                null);

        ColorModel colorModel = new DirectColorModel(24, 0xff0000, 0xff00, 0xff);

        BufferedImage img = new BufferedImage(colorModel, raster, false, null);

        //System.out.println("writing image to " + fileName);
        try {
            ImageIO.write(img, "png", new File(fileName));
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private static void printArray(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(" | ");
            for (int j = 0; j < a[i].length; j++) {
                System.out.print(a[i][j] + " | ");
            }
            System.out.println();
            System.out.println("________________");
            System.out.println();
        }
    }
}
