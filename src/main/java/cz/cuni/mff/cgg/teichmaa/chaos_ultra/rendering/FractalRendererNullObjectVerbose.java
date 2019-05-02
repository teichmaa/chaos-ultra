package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params.RenderingModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.JavaHelpers;

public class FractalRendererNullObjectVerbose implements FractalRenderer {

    public static boolean verbose = true;

    private void printMethodName() {
        if (!verbose) return;
        System.out.println("FractalRendererNullObjectVerbose." + JavaHelpers.getCallingMethodName(1));

    }

    @Override
    public void initializeRendering(OpenGLParams glParams) {
        printMethodName();
    }

    @Override
    public void freeRenderingResources() {
        printMethodName();
    }

    @Override
    public int getWidth() {
        printMethodName();
        return 0;
    }

    @Override
    public int getHeight() {
        printMethodName();
        return 0;
    }

    @Override
    public void launchDebugKernel() {
        printMethodName();
    }

    @Override
    public void renderFast(RenderingModel model) {
        printMethodName();
    }

    @Override
    public void renderQuality(RenderingModel model) {
        printMethodName();
    }

    @Override
    public void close() {
        printMethodName();
    }

    @Override
    public void debugRightBottomPixel() {
        printMethodName();
    }

    @Override
    public void setFractalSpecificParams(String text) {
        printMethodName();
    }

    @Override
    public String getFractalName() {
        printMethodName();
        return "null object fractal";
    }

    @Override
    public FractalRendererState getState() {
        printMethodName();
        return FractalRendererState.notInitialized;
    }
}
