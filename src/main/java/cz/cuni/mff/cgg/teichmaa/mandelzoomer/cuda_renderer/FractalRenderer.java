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
    public static final int FOVEATION_CENTER_RADIUS = 600;//todo zjistit vypoctem
    private static final Pointer NULLPTR = Pointer.to(new byte[0]);

    static {
        CudaHelpers.cudaInit();
    }

    private KernelUnderSampled kernelUndersampled;
    private KernelMain kernelMain;
    //private KernelBlur kernelBlur;
    private KernelCompose kernelCompose;

    private FractalRenderingModule module;

    private int blockDimX = 32;
    private int blockDimY = 32;
    private int paletteLength;

    private CUdeviceptr output2Darray1 = new CUdeviceptr();
    private CUdeviceptr output2Darray2 = new CUdeviceptr();
    private long output2Darray1Pitch;
    private long output2Darray2Pitch;
    private CUdeviceptr randomValues;

    private cudaGraphicsResource outputTextureResource = new cudaGraphicsResource();
    private cudaGraphicsResource paletteTextureResource = new cudaGraphicsResource();
    private cudaStream_t defaultStream = new cudaStream_t();

    public FractalRenderer(FractalRenderingModule module, int outputTextureGLHandle, int outputTextureGLTarget, int paletteTextureGLHandle, int paletteTextureGLTarget, int paletteLength) {
        this.module = module;
        this.paletteLength = paletteLength;
        registerOutputTexture(outputTextureGLHandle, outputTextureGLTarget);
        JCuda.cudaGraphicsGLRegisterImage(paletteTextureResource, paletteTextureGLHandle, paletteTextureGLTarget, cudaGraphicsRegisterFlagsReadOnly);

        kernelUndersampled = module.getKernel(KernelUnderSampled.class);
        kernelUndersampled.setUnderSamplingLevel(4);
        kernelMain = module.getKernel(KernelMain.class);
        kernelCompose = module.getKernel(KernelCompose.class);
        //kernelBlur = module.getKernel(KernelBlur.class);

        output2Darray1Pitch = allocateDevice2DBuffer(kernelMain.getWidth(), kernelMain.getHeight(), output2Darray1);
        output2Darray2Pitch = allocateDevice2DBuffer(kernelMain.getWidth(), kernelMain.getHeight(), output2Darray2);

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
        cuLaunchKernel(kernel.getFunction(),1, 1,kernelParams);
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
     * @return pitch (actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.)
     */
    private long allocateDevice2DBuffer(int width, int height, CUdeviceptr target) {

        /**
         * Pitch = actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
         */
        long pitch;
        long[] pitchptr = new long[1];

        if(target != null){
            cuMemFree(target);
        }else{
            target = new CUdeviceptr();
        }

        if(width * height == 0)
            return 0;

        cuMemAllocPitch(target, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);

        pitch = pitchptr[0];
        if (pitch <= 0) {
            throw new CudaException("cuMemAllocPitch returned pitch with value 0 (or less)");
        }

        if (pitch > Integer.MAX_VALUE) {
            //this would mean an array with length bigger that Integer.MAX_VALUE and this is not supported by Java
            System.err.println("Warning: allocateDevice2DBuffer: pitch > Integer.MAX_VALUE");
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
        kernelMain.setWidth(width);
        kernelUndersampled.setWidth(width);
        kernelMain.setHeight(height);
        kernelUndersampled.setHeight(height);
        randomSamplesInit();
        registerOutputTexture(outputTextureGLhandle, GLtarget);
        output2Darray1Pitch = allocateDevice2DBuffer(width, height, output2Darray1);
        output2Darray2Pitch = allocateDevice2DBuffer(width, height, output2Darray2);
    }

    public int getWidth() {
        return kernelMain.getWidth();
    }

    public int getHeight() {
        return kernelMain.getHeight();
    }

    /**
     *
     */
    public void launchFastKernel(int focusx, int focusy) {
        kernelMain.setFocus(focusx, focusy);
        kernelMain.setRenderRadius(FOVEATION_CENTER_RADIUS);
        //todo nejak spustit ty prvni dva najednou a pak pockat?
        launchRenderingKernel(false, kernelMain, output2Darray1, output2Darray1Pitch);
        launchRenderingKernel(false, kernelUndersampled, output2Darray2, output2Darray2Pitch);
        launchDrawingKernel(false, kernelCompose);
    }

    public void launchQualityKernel(){
        kernelMain.setRenderRadiusToMax();
        kernelMain.setFocusDefault();
        launchRenderingKernel(false, kernelMain, output2Darray1, output2Darray1Pitch);
        launchDrawingKernel(false, kernelCompose);
    }

    private void launchRenderingKernel(boolean async, RenderingKernel kernel, CUdeviceptr output, long outputPitch){
        int width = kernel.getWidth();
        int height = kernel.getHeight();

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        NativePointerObject[] kernelParamsArr = kernel.getKernelParams();
        {
            kernelParamsArr[kernel.PARAM_IDX_2DARR_OUT] = Pointer.to(output);
            kernelParamsArr[kernel.PARAM_IDX_2DARR_OUT_PITCH] = Pointer.to(new long[]{outputPitch});
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
                    gridDimX, gridDimY,
                    kernelParams
            );
            cuCtxSynchronize();
            if (!async)
                cuCtxSynchronize();
        } catch (CudaException e) {
            System.err.println("Error just after launching a kernel:");
            System.err.println(e);
        }
    }

    private void launchDrawingKernel(boolean async, KernelCompose kernel) {
        long start = System.currentTimeMillis();

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
                    int gridDimX = getWidth() / blockDimX;
                    int gridDimY = getHeight() / blockDimY;

                    NativePointerObject[] composeParamsArr = kernel.getKernelParams();
                    {
                        composeParamsArr[kernel.PARAM_IDX_WIDTH] = Pointer.to(new int[]{getWidth()});
                        composeParamsArr[kernel.PARAM_IDX_HEIGHT] = Pointer.to(new int[]{getHeight()});
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_OUT] = Pointer.to(surfaceOutput);
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN] = Pointer.to(output2Darray1);
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG] = Pointer.to(output2Darray2);
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN_PITCH] = Pointer.to(new long[]{output2Darray1Pitch});
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG_PITCH] = Pointer.to(new long[]{output2Darray2Pitch});
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_PALETTE] = Pointer.to(surfacePalette);
                        composeParamsArr[kernel.PARAM_IDX_PALETTE_LENGTH] = Pointer.to(new int[]{paletteLength});
                        composeParamsArr[kernel.PARAM_IDX_DWELL] = Pointer.to(new int[]{getDwell()});
                        composeParamsArr[kernel.PARAM_IDX_MAIN_RADIUS] = Pointer.to(new int[]{kernelMain.getRenderRadius()});
                        composeParamsArr[kernel.PARAM_IDX_FOCUS_X] = Pointer.to(new int[]{kernelMain.getFocusx()});
                        composeParamsArr[kernel.PARAM_IDX_FOCUS_Y] = Pointer.to(new int[]{kernelMain.getFocusy()});
                    }

                    if (gridDimX <= 0 || gridDimY <= 0) return;
                    if (gridDimX > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM * blockDimX);
                    }
                    if (gridDimY > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM * blockDimY);
                    }
                    try {
                        cuLaunchKernel(kernel.getFunction(),
                                gridDimX, gridDimY,
                                Pointer.to(composeParamsArr)
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

    private void cuLaunchKernel(CUfunction kernelFunction, int gridDimX, int gridDimY, Pointer kernelParams){
        JCudaDriver.cuLaunchKernel(kernelFunction,
                gridDimX, gridDimY, 1,
                blockDimX, blockDimY, 1,
                0, null,           // Shared memory size and defaultStream
                kernelParams, null // Kernel- and extra parameters
        );
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
        kernelMain.setAdaptiveSS(adaptiveSS);
    }

    public void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        kernelMain.setVisualiseAdaptiveSS(visualiseAdaptiveSS);
    }

    public void setSuperSamplingLevel(int supSampLvl) {
        kernelMain.setSuperSamplingLevel(supSampLvl);
        //randomSamplesInit();
    }

    public void setDwell(int dwell) {
        kernelMain.setDwell(dwell);
        kernelUndersampled.setDwell(dwell);
    }

    public int getSuperSamplingLevel() {
        return kernelMain.getSuperSamplingLevel();
    }

    public int getDwell() {
        return kernelMain.getDwell();
    }

    public void setBounds(float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        kernelMain.setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        kernelUndersampled.setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

}
