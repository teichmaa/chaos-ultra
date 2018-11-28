package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

import java.io.Closeable;

public interface FractalRenderer extends Closeable {
    int CUDA_MAX_GRID_DIM = 65536 - 1;
    int SUPER_SAMPLING_MAX_LEVEL = 256;

    //todo dokumentacni komentar
    void registerOutputTexture(int outputTextureGLhandle, int GLtarget);

    void unregisterOutputTexture();

    void resize(int width, int height, int outputTextureGLhandle, int GLtarget);

    int getWidth();

    int getHeight();

    void launchDebugKernel();

    void renderFast(Point2DInt focus, boolean isZooming);

    void renderQuality();

    @Override
    void close();

    void setAdaptiveSS(boolean adaptiveSS);

    void setVisualiseSampleCount(boolean visualiseAdaptiveSS);

    void setSuperSamplingLevel(int supSampLvl);

    void setMaxIterations(int MaxIterations);

    int getSuperSamplingLevel();

    int getDwell();

    void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y);

    void setUseFoveation(boolean value);

    void setUseSampleReuse(boolean value);

    void debugRightBottomPixel();

    default FloatPrecision getPrecision() {return FloatPrecision.singlePrecision; }
}
