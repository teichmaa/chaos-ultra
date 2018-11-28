package cz.cuni.mff.cgg.teichmaa.chaos_ultra.fractal;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Helpers;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

public class FractalRendererNullObjectVerbose implements FractalRenderer {

    public static boolean verbose = true;
    private void printMethodName(){
        if(!verbose) return;
        System.out.println("FractalRendererNullObjectVerbose." + Helpers.getCallingMethodName(1));

    }

    @Override
    public void registerOutputTexture(int outputTextureGLhandle, int GLtarget) {
        printMethodName();
    }

    @Override
    public void unregisterOutputTexture() {
        printMethodName();
    }

    @Override
    public void resize(int width, int height, int outputTextureGLhandle, int GLtarget) {
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
    public void setAdaptiveSS(boolean adaptiveSS) {
        printMethodName();
    }

    @Override
    public void setVisualiseSampleCount(boolean visualiseAdaptiveSS) {
        printMethodName();
    }

    @Override
    public void setSuperSamplingLevel(int supSampLvl) {
        printMethodName();
    }

    @Override
    public void setMaxIterations(int MaxIterations) {
        printMethodName();
    }

    @Override
    public int getSuperSamplingLevel() {
        printMethodName();
        return 0;
    }

    @Override
    public int getDwell() {
        printMethodName();
        return 0;
    }

    @Override
    public void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {
        printMethodName();
    }

    @Override
    public void setUseFoveation(boolean value) {
        printMethodName();
    }

    @Override
    public void setUseSampleReuse(boolean value) {
        printMethodName();
    }

    @Override
    public void debugRightBottomPixel() {
        printMethodName();
    }

}
