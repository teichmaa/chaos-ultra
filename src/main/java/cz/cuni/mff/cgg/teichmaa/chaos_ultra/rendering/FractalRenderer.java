package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import java.io.Closeable;

public interface FractalRenderer extends Closeable {
    int SUPER_SAMPLING_MAX_LEVEL = 256;

    //todo dokumentacni komentar
    //using this call-me-approach rather than unregistering the resource every time a frame is rendered is for performance reasons, see https://devtalk.nvidia.com/default/topic/747242/cuda-opengl-interop-performance/
    //may be called only when a corresponding OpenGL context is active (e.g. during GLEventListener events). Otherwise cudaErrorInvalidGraphicsContext or some other errors are to expect.
    void initializeRendering(GLParams glParams);

    void freeRenderingResources();

    FractalRendererState getState();

    int getWidth();

    int getHeight();

    void launchDebugKernel();

    void renderFast(Model model);

    void renderQuality(Model model);

    @Override
    void close();

    void debugRightBottomPixel();

    void setFractalSpecificParams(String text);

    String getFractalName();
}
