package cz.cuni.mff.cgg.teichmaa.chaos_ultra.fractal;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

public class FractalRendererNullObject implements FractalRenderer {
    //todo, tohle by cele melo  byt designovane jinak:
    // fractal renderer je interface, ve vlastnim baliku (fractal), v nem je i tahle implementace
    // cuda balik dodava implementaci pro CudaFractalRenderer a Module
    // novy CudaFractalRenderer se dela pomoci builderu, rekne se mu, ktery fraktal a ktera platforma (CUDA) a on to podle toho zkonstruuje
    // nekde v baliku fractal tedy bude nejaky katalog fraktalu a platforem

    @Override
    public void registerOutputTexture(int outputTextureGLhandle, int GLtarget) {

    }

    @Override
    public void unregisterOutputTexture() {

    }

    @Override
    public void resize(int width, int height, int outputTextureGLhandle, int GLtarget) {

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
    public void setAdaptiveSS(boolean adaptiveSS) {

    }

    @Override
    public void setVisualiseSampleCount(boolean visualiseAdaptiveSS) {

    }

    @Override
    public void setSuperSamplingLevel(int supSampLvl) {

    }

    @Override
    public void setMaxIterations(int MaxIterations) {

    }

    @Override
    public int getSuperSamplingLevel() {
        return 0;
    }

    @Override
    public int getDwell() {
        return 0;
    }

    @Override
    public void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {

    }

    @Override
    public void setUseFoveation(boolean value) {

    }

    @Override
    public void setUseSampleReuse(boolean value) {

    }

    @Override
    public void debugRightBottomPixel() {

    }

}
