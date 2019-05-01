package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

import java.io.Closeable;

public interface FractalRenderer extends Closeable {
    int SUPER_SAMPLING_MAX_LEVEL = 256;

    //todo dokumentacni komentar
    //using this call-me-approach rather than unregistering the resource every time a frame is rendered is for performance reasons, see https://devtalk.nvidia.com/default/topic/747242/cuda-opengl-interop-performance/
    void initializeRendering(OpenGLParams glParams);

    void freeRenderingResources();

    FractalRendererState getState();

    int getWidth();

    int getHeight();

    void launchDebugKernel();

    void renderFast(Point2DInt focus, boolean isZooming);

    void renderQuality();

    @Override
    void close();

    void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y);

    void debugRightBottomPixel();

    void bindParamsTo(ChaosUltraRenderingParams params);

    void setFractalSpecificParams(String text);

    String getFractalName();
}
