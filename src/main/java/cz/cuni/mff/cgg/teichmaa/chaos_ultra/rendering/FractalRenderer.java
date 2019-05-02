package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params.RenderingModel;

import java.io.Closeable;

public interface FractalRenderer extends Closeable {
    int SUPER_SAMPLING_MAX_LEVEL = 256;

    //todo dokumentacni komentar
    //using this call-me-approach rather than unregistering the resource every time a frame is rendered is for performance reasons, see https://devtalk.nvidia.com/default/topic/747242/cuda-opengl-interop-performance/
    //may be called only when a corresponding OpenGL context is active (e.g. during GLEventListener events). Otherwise cudaErrorInvalidGraphicsContext or some other errors are to expect.
    void initializeRendering(OpenGLParams glParams);

    void freeRenderingResources();

    FractalRendererState getState();

    int getWidth();

    int getHeight();

    void launchDebugKernel();

    void renderFast(RenderingModel model);

    void renderQuality(RenderingModel model);

    @Override
    void close();

//    void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y);

    void debugRightBottomPixel();

//    void bindParamsTo(RenderingModel params);

    void setFractalSpecificParams(String text);

    String getFractalName();
}
