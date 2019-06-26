package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRenderer;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRendererException;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRendererState;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.*;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.JavaHelpers;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.SimpleLogger;
import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.*;

import java.nio.IntBuffer;
import java.security.InvalidParameterException;
import java.util.function.Consumer;

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuMemcpy2D;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly;
import static jcuda.runtime.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard;
import static jcuda.runtime.cudaResourceType.cudaResourceTypeArray;

/**
 *
 * Lifecycle:  notInitialized      --- initializeRendering() --->   readyToRender
 *             readyToRender    --- freeRenderingResources() --->   notInitialized
 */
public class CudaFractalRenderer implements FractalRenderer {

    private static final int CUDA_MAX_GRID_DIM = 65536 - 1;
    private static final Pointer NULLPTR = CudaHelpers.pointerTo(0);
    private static final int BLOCK_DIM_X = 32;
    private static final int BLOCK_DIM_Y = 32;
    private static final int DEFAULT_UNDER_SAMPLING_LEVEL = 4;
    private static final int SIZEOF_PIXEL_INFO_T = 4;
    /**
     * Name of the "visualise sample count" cuda constant in the cuda module
     */
    private static final String VISUALIZE_SAMPLE_COUNT_CONSTANT_NAME = "VISUALIZE_SAMPLE_COUNT";

    static {
        CudaHelpers.cudaInit();
    }

    private KernelUnderSampled kernelUndersampled;
    private KernelMainFloat kernelMainFloat;
    private KernelMainDouble kernelMainDouble;
    private KernelAdvancedFloat kernelAdvancedFloat;
    private KernelAdvancedDouble kernelAdvancedDouble;
    private KernelCompose kernelCompose;

    private DeviceMemoryDoubleBuffer2D memory = new DeviceMemoryDoubleBuffer2D(SIZEOF_PIXEL_INFO_T);
    private FractalRenderingModule module;
    private cudaStream_t defaultStream = new cudaStream_t();

    private GLParams glParams;
    private cudaGraphicsResource outputTextureResource = new cudaGraphicsResource();
    private cudaGraphicsResource paletteTextureResource = new cudaGraphicsResource();

    //todo dokumentacni komentar k metode
    public CudaFractalRenderer(FractalRenderingModule module) {
        this.module = module;

        kernelUndersampled = module.getKernel(KernelUnderSampled.class);
        kernelUndersampled.setUnderSamplingLevel(DEFAULT_UNDER_SAMPLING_LEVEL);
        kernelMainFloat = module.getKernel(KernelMainFloat.class);
        kernelMainDouble = module.getKernel(KernelMainDouble.class);
        kernelCompose = module.getKernel(KernelCompose.class);
        kernelAdvancedFloat = module.getKernel(KernelAdvancedFloat.class);
        kernelAdvancedDouble = module.getKernel(KernelAdvancedDouble.class);

        memory.reallocate(getWidth(), getHeight());
    }

    private FractalRendererState state = FractalRendererState.notInitialized;

    /**
     * @throws jcuda.CudaException containing cudaErrorInvalidGraphicsContext when OpenGL is not active
     */
    @Override
    public void initializeRendering(GLParams glParams) {
        if(state == FractalRendererState.readyToRender) throw new IllegalStateException("Already initialized.");

        this.glParams = glParams;
        GLTexture outputTexture = glParams.getOutput();
        int width = outputTexture.getWidth();
        int height = outputTexture.getHeight();

        onAllRenderingKernels(k -> k.setOutputSize(width, height));
        memory.reallocate(width, height);

        //documentation: http://www.jcuda.org/jcuda/doc/jcuda/runtime/JCuda.html#cudaGraphicsGLRegisterImage(jcuda.runtime.cudaGraphicsResource,%20int,%20int,%20int)
        JCuda.cudaGraphicsGLRegisterImage(outputTextureResource, outputTexture.getHandle().getValue(), outputTexture.getTarget(), cudaGraphicsRegisterFlagsWriteDiscard);
        JCuda.cudaGraphicsGLRegisterImage(paletteTextureResource, glParams.getPalette().getHandle().getValue(), glParams.getPalette().getTarget(), cudaGraphicsRegisterFlagsReadOnly);

        state = FractalRendererState.readyToRender;
    }

    @Override
    public void freeRenderingResources() {
        if(state == FractalRendererState.notInitialized) throw new IllegalStateException("Already free.");

        memory.memoryFree();
        glParams = null;
        JCuda.cudaGraphicsUnregisterResource(outputTextureResource);
        JCuda.cudaGraphicsUnregisterResource(paletteTextureResource);
        state = FractalRendererState.notInitialized;
    }

    @Override
    public FractalRendererState getState() {
        return state;
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

    @Override
    public int getWidth() {
        return kernelMainFloat.getWidth();
    }

    @Override
    public int getHeight() {
        return kernelMainFloat.getHeight();
    }

    @Override
    public void launchDebugKernel() {
        CudaKernel k = module.getKernel(KernelDebug.class);
        cuLaunchKernel(k.getFunction(),
                1, 1,
                Pointer.to(k.getKernelParams())
        );
        JCudaDriver.cuCtxSynchronize();
    }

    private RenderingModel lastRendering;

    @Override
    public void renderFast(RenderingModel model) {
        if(state != FractalRendererState.readyToRender) throw new IllegalStateException("Renderer has to be initialized first");

        setModuleConstants(model);

        if (model.isSampleReuseCacheDirty() || memory.isPrimary2DBufferDirty() || lastRendering == null) {
            //if there is nothing to reuse, then create it
            renderQuality(model);
            return;
        }

        updateFloatPrecision(model);
        KernelAdvanced k = getFloatOrDoubleKernel(model.getFloatingPointPrecision(), kernelAdvancedFloat, kernelAdvancedDouble);

        k.setOriginSegment(lastRendering.getPlaneSegment());
        k.setParamsFromModel(model);
        k.setInput(memory.getPrimary2DBuffer(), memory.getPrimary2DBufferPitch());
        k.setOutput(memory.getSecondary2DBuffer(), memory.getSecondary2DBufferPitch());
        kernelCompose.setParamsFromModel(model);

        launchRenderingKernel(false, k, model);
        memory.switch2DBuffers();
        launchDrawingKernel(false, kernelCompose, model);

        lastRendering = model.copy();
    }

    @Override
    public void renderQuality(RenderingModel model) {
        if(state != FractalRendererState.readyToRender) throw new IllegalStateException("Renderer has to be initialized first");

        setModuleConstants(model);

        updateFloatPrecision(model);
        KernelMain kernelMain = getFloatOrDoubleKernel(model.getFloatingPointPrecision(), kernelMainFloat, kernelMainDouble);

        memory.resetBufferOrder();
        kernelMain.setParamsFromModel(model);
        kernelMain.setOutput(memory.getPrimary2DBuffer(), memory.getPrimary2DBufferPitch());
        kernelCompose.setParamsFromModel(model);

        launchRenderingKernel(false, kernelMain, model);
        launchDrawingKernel(false, kernelCompose, model);

        lastRendering = model.copy();
        memory.setPrimary2DBufferDirty(false);
        model.setSampleReuseCacheDirty(false);
    }

    private <T extends RenderingKernel> T getFloatOrDoubleKernel(FloatPrecision precision, T kernelFloat, T kernelDouble) {
        T k;
        switch (precision) {
            case doublePrecision:
                k = kernelDouble;
                break;
            case singlePrecision:
                k = kernelFloat;
                break;
            case tooBig:
                k = kernelDouble;
                SimpleLogger.get().logRenderingInfo("CudaFractalRenderer: tooBig precision with render()");
                break;
            default:
                throw new IllegalStateException("Precision not supported: " + precision);
        }
        return k;
    }

    private void setModuleConstants(RenderingModel model) {
        if (lastRendering == null ||
                model.isVisualiseSampleCount() != lastRendering.isVisualiseSampleCount()
        ) {
            module.writeToConstantMemory(VISUALIZE_SAMPLE_COUNT_CONSTANT_NAME, model.isVisualiseSampleCount());
        }
    }

    private void launchRenderingKernel(boolean async, RenderingKernel kernel, PublicErrorLogger logger) {
        int width = kernel.getWidth();
        int height = kernel.getHeight();

        // Following the Jcuda API, kernel heuristicsParams is a pointer to an array of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(kernel.getKernelParams());
        CUfunction kernelFunction = kernel.getFunction();
        int gridDimX = (width + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
        int gridDimY = (height + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;



        if (gridDimX <= 0 || gridDimY <= 0) return;
        if (gridDimX > CUDA_MAX_GRID_DIM) {
            throw new FractalRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM * BLOCK_DIM_X);
        }
        if (gridDimY > CUDA_MAX_GRID_DIM) {
            throw new FractalRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM * BLOCK_DIM_Y);
        }
        try {
            cuLaunchKernel(kernelFunction,
                    gridDimX, gridDimY,
                    kernelParams
            );
            //TODO wtf, one ctx synchronisation is enough, no?
            cuCtxSynchronize();
            if (!async)
                cuCtxSynchronize();
        } catch (CudaException e) {
            logger.logError("Error just after launching a kernel:" + e.getMessage());
            if (JavaHelpers.isDebugMode()) {
                SimpleLogger.get().error(e.toString());
                throw e;
            }
        }
    }

    /**
     * @param async
     * @param kernel
     */
    private void launchDrawingKernel(boolean async, KernelCompose kernel, PublicErrorLogger logger) {
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
                    int gridDimX = (getWidth() + BLOCK_DIM_X - 1) / BLOCK_DIM_X;
                    int gridDimY = (getHeight() + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;

                    // Set up the kernel parameters: A pointer to an array
                    // of pointers which point to the actual values.
                    NativePointerObject[] composeParamsArr = kernel.getKernelParams();
                    {
                        //TODO this pattern has to be changed and updated to java style
                        composeParamsArr[kernel.PARAM_IDX_WIDTH] = CudaHelpers.pointerTo(getWidth());
                        composeParamsArr[kernel.PARAM_IDX_HEIGHT] = CudaHelpers.pointerTo(getHeight());
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_OUT] = Pointer.to(surfaceOutput);
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN] = Pointer.to(memory.getPrimary2DBuffer());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG] = Pointer.to(memory.getSecondary2DBuffer());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_MAIN_PITCH] = CudaHelpers.pointerTo(memory.getPrimary2DBufferPitch());
                        composeParamsArr[kernel.PARAM_IDX_INPUT_BCG_PITCH] = CudaHelpers.pointerTo(memory.getSecondary2DBufferPitch());
                        composeParamsArr[kernel.PARAM_IDX_SURFACE_PALETTE] = Pointer.to(surfacePalette);
                        composeParamsArr[kernel.PARAM_IDX_PALETTE_LENGTH] = CudaHelpers.pointerTo(glParams.getPaletteLength());
                    }

                    if (gridDimX <= 0 || gridDimY <= 0) return;
                    if (gridDimX > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: width must be smaller than " + CUDA_MAX_GRID_DIM * BLOCK_DIM_X);
                    }
                    if (gridDimY > CUDA_MAX_GRID_DIM) {
                        throw new FractalRendererException("Unsupported input parameter: height must be smaller than " + CUDA_MAX_GRID_DIM * BLOCK_DIM_Y);
                    }
                    try {
                        cuLaunchKernel(kernel.getFunction(),
                                gridDimX, gridDimY,
                                Pointer.to(composeParamsArr)
                        );
                        if (!async)
                            cuCtxSynchronize();
                    } catch (CudaException e) {
                        logger.logError("Error just after launching a kernel:" + e.getMessage());
                        if (JavaHelpers.isDebugMode()) {
                            SimpleLogger.get().error(e.toString());
                            throw e;
                        }
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
            logger.logError("Error during kernel launch preparation:" + e.getMessage());
            if (JavaHelpers.isDebugMode()) {
                SimpleLogger.get().error(e.toString());
                throw e;
            }
        }

        long end = System.currentTimeMillis();
        //System.out.println("Kernel " + kernel.getMainFunctionName() + " launched and finished in " + (end - start) + "ms");
    }

    /**
     * just a syntactic sugar - just call the JCudaDriver.cuLaunchKernel with some more parameter's default values
     *
     * @param kernelParams
     */
    private void cuLaunchKernel(CUfunction kernelFunction, int gridDimX, int gridDimY, Pointer kernelParams) {
        JCudaDriver.cuLaunchKernel(kernelFunction,
                gridDimX, gridDimY, 1,
                BLOCK_DIM_X, BLOCK_DIM_Y, 1,
                0, null,           // Shared memory size and defaultStream
                kernelParams, null // Kernel- and extra parameters
        );
    }

    @Override
    public void close() {
        if(state == FractalRendererState.readyToRender)
            freeRenderingResources();
        memory.close();
        module.close();
    }

    @Override
    public void setFractalCustomParams(String text) {
        module.setFractalCustomParameters(text);
    }

    private void onAllRenderingKernels(Consumer<RenderingKernel> c) {
        onAllMainKernels(c);
        c.accept(kernelUndersampled);
    }

    private void onAllMainKernels(Consumer<? super KernelMain> c) {
        onAllAdvancedKernels(c);
        c.accept(kernelMainFloat);
        c.accept(kernelMainDouble);
    }

    private void onAllAdvancedKernels(Consumer<? super KernelAdvanced> c) {
        c.accept(kernelAdvancedFloat);
        c.accept(kernelAdvancedDouble);
    }

    private void updateFloatPrecision(RenderingModel model) {
        kernelMainDouble.setParamsFromModel(model);
        FloatPrecision precison = FloatPrecision.singlePrecision;
        if (kernelMainDouble.isSegmentBoundsAtFloatLimit()) {
            precison = FloatPrecision.doublePrecision;
        }
        if (kernelMainDouble.isSegmentBoundsAtDoubleLimit()) {
            precison = FloatPrecision.tooBig;
        }
        model.setFloatingPointPrecision(precison);
    }

    @Override
    public String getFractalName() {
        return module.getFractalName();
    }

    @Override
    public void supplyDefaultValues(DefaultFractalModel model) {
        module.supplyDefaultValues(model);
    }
}
