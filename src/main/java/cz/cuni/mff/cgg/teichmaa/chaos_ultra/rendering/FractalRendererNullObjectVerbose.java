package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Helpers;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

public class FractalRendererNullObjectVerbose implements FractalRenderer {

    public static boolean verbose = true;

    private void printMethodName() {
        if (!verbose) return;
        System.out.println("FractalRendererNullObjectVerbose." + Helpers.getCallingMethodName(1));

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
    public void renderFast(Point2DInt focus, boolean isZooming) {
        printMethodName();
    }

    @Override
    public void renderQuality() {
        printMethodName();
    }

    @Override
    public void close() {
        printMethodName();
    }

    @Override
    public void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {
        printMethodName();
    }

    @Override
    public void debugRightBottomPixel() {
        printMethodName();
    }

    @Override
    public void bindParamsTo(ChaosUltraRenderingParams params) {
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
