package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

public class FractalRendererNullObject implements FractalRenderer {

    @Override
    public void initializeRendering(OpenGLParams glParams) {

    }

    @Override
    public void freeRenderingResources() {

    }

    @Override
    public int getWidth() {
        return 0;
    }

    @Override
    public int getHeight() {
        return 0;
    }

    @Override
    public void launchDebugKernel() {

    }

    @Override
    public void renderFast(Point2DInt focus, boolean isZooming) {

    }

    @Override
    public void renderQuality() {

    }

    @Override
    public void close() {

    }

    @Override
    public void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {

    }

    @Override
    public void debugRightBottomPixel() {

    }

    @Override
    public void bindParamsTo(ChaosUltraRenderingParams params) {

    }

    @Override
    public void setFractalSpecificParams(String text) {

    }

    @Override
    public String getFractalName() {
        return "null object fractal";
    }

    @Override
    public FractalRendererState getState() {
        return FractalRendererState.notInitialized;
    }
}
