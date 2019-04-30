package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

import java.io.Closeable;

public interface FractalRenderer extends Closeable {
    int SUPER_SAMPLING_MAX_LEVEL = 256;

    //todo dokumentacni komentar
    void registerOutputTexture(OpenGLTexture output);

    void unregisterOutputTexture();

    void resize(int width, int height, OpenGLTexture output);

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
