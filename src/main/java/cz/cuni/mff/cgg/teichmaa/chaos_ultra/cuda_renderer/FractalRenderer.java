package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;
import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.io.Closeable;
import java.nio.IntBuffer;
import java.security.InvalidParameterException;
import java.util.function.Consumer;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

public class FractalRenderer implements Closeable {

    public static final int CUDA_MAX_GRID_DIM = 65536 - 1;
    public static final int SUPER_SAMPLING_MAX_LEVEL = 256;
    private static final Pointer NULLPTR = CudaHelpers.pointerTo(0);

    static {
        CudaHelpers.cudaInit();
    }

    private KernelUnderSampled kernelUndersampled;
    private KernelMainFloat kernelMainFloat;
    private KernelMainDouble kernelMainDouble;
    private KernelAdvancedFloat kernelAdvancedFloat;
    private KernelAdvancedDouble kernelAdvancedDouble;
    private KernelCompose kernelCompose;

    private DeviceMemory memory = new DeviceMemory();
    private FractalRenderingModule module;

    private int blockDimX = 32;
    private int blockDimY = 32;
    private int paletteLength;

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
        kernelMainFloat = module.getKernel(KernelMainFloat.class);
        kernelMainDouble = module.getKernel(KernelMainDouble.class);
        kernelCompose = module.getKernel(KernelCompose.class);
        kernelAdvancedFloat = module.getKernel(KernelAdvancedFloat.class);
        kernelAdvancedDouble = module.getKernel(KernelAdvancedDouble.class);

        memory.reallocatePrimary2DBuffer(getWidth(), getHeight());
        memory.reallocateSecondary2DBuffer(getWidth(), getHeight());

        //moduleInit();
    }

    private void moduleInit() {

        KernelInit kernel = module.getKernel(KernelInit.class);

        Pointer kernelParams = Pointer.to(kernel.getKernelParams());
        cuLaunchKernel(kernel.getFunction(), 1, 1, kernelParams);
        cuCtxSynchronize();
    }

    public void registerOutputTexture(int outputTextureGLhandle, int GLtarget) {
        //documentation: http://www.jcuda.org/jcuda/doc/jcuda/runtime/JCuda.html#cudaGraphicsGLRegisterImage(jcuda.runtime.cudaGraphicsResource,%20int,%20int,%20int)
        jcuda.runtime.JCuda.cudaGraphicsGLRegisterImage(outputTextureResource, outputTextureGLhandle, GLtarget, cudaGraphicsRegisterFlagsWriteDiscard);
    }

    public void unregisterOutputTexture() {
        jcuda.runtime.JCuda.cudaGraphicsUnregisterResource(outputTextureResource);
    }


    private void copy2DFromDevToHost(IntBuffer hostOut, int width, int height, long pitch, CUdeviceptr deviceOut) {
        if (hostOut.capacity() < width * height)
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
        onAllRenderingKernels(k -> k.setOutputSize(width, height));
        registerOutputTexture(outputTextureGLhandle, GLtarget);
        memory.reallocatePrimary2DBuffer(width, height);
        memory.reallocateSecondary2DBuffer(width, height);
    }

    public int getWidth() {
        return kernelMainFloat.getWidth();
    }

    public int getHeight() {
        return kernelMainFloat.getHeight();
    }

    public void launchDebugKernel() {

        CudaKernel k = module.getKernel(KernelDebug.class);
        cuLaunchKernel(k.getFunction(),
                1, 1,
                Pointer.to(k.getKernelParams())
        );
        JCudaDriver.cuCtxSynchronize();
    }

    RenderingKernelParamsInfo lastRendering = new RenderingKernelParamsInfo();

    public void renderFast(Point2DInt focus, boolean isZooming) {
        if (memory.isPrimary2DBufferUnusable()) {
            //if there is nothing to reuse, then create it
            renderQuality();
            return;
        }
        KernelAdvanced k = kernelAdvancedFloat.isBoundsAtFloatLimit() ? kernelAdvancedDouble : kernelAdvancedFloat;
        k.setOriginBounds(lastRendering.left_bottom_x, lastRendering.left_bottom_y, lastRendering.right_top_x, lastRendering.right_top_y);
        k.setFocus(focus.getX(), focus.getY());
        k.setInput(memory.getPrimary2DBuffer(), memory.getPrimary2DBufferPitch());
        k.setOutput(memory.getSecondary2DBuffer(), memory.getSecondary2DBufferPitch());
        k.setIsZooming(isZooming);

        launchRenderingKernel(false, k);
        memory.switch2DBuffers();
        lastRendering.setFrom(k);
        launchDrawingKernel(false, kernelCompose);
    }

    public void renderQuality() {
        KernelMain kernelMain = kernelMainFloat.isBoundsAtFloatLimit() ? kernelMainDouble : kernelMainFloat;
        memory.resetBufferSwitch();
        kernelMain.setOutput(memory.getPrimary2DBuffer(), memory.getPrimary2DBufferPitch());
        launchRenderingKernel(false, kernelMain);
        launchDrawingKernel(false, kernelCompose);
        lastRendering.setFrom(kernelMain);
        memory.setPrimary2DBufferUnusable(false);
    }

    private void launchRenderingKernel(boolean async, RenderingKernel kernel) {
        int width = kernel.getWidth();
        int height = kernel.getHeight();

        // Following the Jcuda API, kerenel params is a pointer to an array of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(kernel.getKernelParams());
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

    /**
     * @param async
     * @param kernel
     */
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
                        composeParamsArr[kernel.PARAM_IDX_WIDTH] = CudaHelpers.pointerTo(getWidth());
                        composeParamsArr[kernel.PARAM_IDX_HEIGHT] = CudaHelpers.pointerTo(getHeight());
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_OUT] = Pointer.to(surfaceOutput);
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN] = Pointer.to(memory.getPrimary2DBuffer());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG] = Pointer.to(memory.getSecondary2DBuffer());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN_PITCH] = CudaHelpers.pointerTo(memory.getPrimary2DBufferPitch());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG_PITCH] = CudaHelpers.pointerTo(memory.getSecondary2DBufferPitch());
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_PALETTE] = Pointer.to(surfacePalette);
                        composeParamsArr[kernel.PARAM_IDX_PALETTE_LENGTH] = CudaHelpers.pointerTo(paletteLength);
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

    private void cuLaunchKernel(CUfunction kernelFunction, int gridDimX, int gridDimY, Pointer kernelParams) {
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
        memory.close();
        module.close();
    }

    public void setAdaptiveSS(boolean adaptiveSS) {
        onAllMainKernels(k -> k.setAdaptiveSS(adaptiveSS));
    }

    public void setVisualiseSampleCount(boolean visualiseAdaptiveSS) {
        if (kernelAdvancedFloat.getVisualiseSampleCount() == true &&
                visualiseAdaptiveSS == false) //when switching from "visualiseAdaptiveSS" mode back to normal, don't reuse the texture
            memory.setPrimary2DBufferUnusable(true);
        onAllMainKernels(k -> k.setVisualiseSampleCount(visualiseAdaptiveSS));
    }

    public void setSuperSamplingLevel(int supSampLvl) {
        onAllMainKernels(k -> k.setSuperSamplingLevel(supSampLvl));
    }

    public void setMaxIterations(int MaxIterations) {
        onAllRenderingKernels(k -> k.setMaxIterations(MaxIterations));
    }

    public int getSuperSamplingLevel() {
        return kernelMainFloat.getSuperSamplingLevel();
    }

    public int getDwell() {
        return kernelMainFloat.getMaxIterations();
    }

    public void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {
        onAllRenderingKernels(k -> k.setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y));
    }

    public void setUseFoveation(boolean value) {
        kernelAdvancedFloat.setUseFoveation(value);
        kernelAdvancedDouble.setUseFoveation(value);
    }

    public void setUseSampleReuse(boolean value) {
        kernelAdvancedFloat.setUseSampleReuse(value);
        kernelAdvancedDouble.setUseSampleReuse(value);
    }

    void onAllRenderingKernels(Consumer<RenderingKernel> c) {
        onAllMainKernels(c);
        c.accept(kernelUndersampled);
    }

    void onAllMainKernels(Consumer<? super KernelMain> c ){
        c.accept(kernelMainFloat);
        c.accept(kernelMainDouble);
        c.accept(kernelAdvancedFloat);
        c.accept(kernelAdvancedDouble);
    }

    public void debugRightBottomPixel() {
        int w = getWidth();
        int h = getHeight();
        memory.resetBufferSwitch();
        //launchReuseSamplesKernel();
        memory.resetBufferSwitch();
        IntBuffer b = IntBuffer.allocate(w * h);
        copy2DFromDevToHost(b, w, h, memory.getPrimary2DBufferPitch(), memory.getPrimary2DBuffer());
        System.out.println("b[w,h]:\t" + b.get(w * h - 1));
        int breakpoit = 0;
    }

    public FloatPrecision getPrecision() {
        return kernelMainDouble.isBoundsAtFloatLimit() ? FloatPrecision.doublePrecision : FloatPrecision.singlePrecision;
    }
}
