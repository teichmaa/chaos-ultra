package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params.RenderingModel;

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
    public void renderFast(RenderingModel model) {

    }

    @Override
    public void renderQuality(RenderingModel model) {

    }

    @Override
    public void close() {

    }

    @Override
    public void debugRightBottomPixel() {

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
