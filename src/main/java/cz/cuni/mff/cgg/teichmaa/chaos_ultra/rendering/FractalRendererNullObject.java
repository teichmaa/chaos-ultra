package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DInt;

public class FractalRendererNullObject implements FractalRenderer {
    //todo, tohle by cele melo  byt designovane jinak:
    // rendering renderer je interface, ve vlastnim baliku (rendering), v nem je i tahle implementace
    // cuda balik dodava implementaci pro CudaFractalRenderer a Module
    // novy CudaFractalRenderer se dela pomoci builderu, rekne se mu, ktery fraktal a ktera platforma (CUDA) a on to podle toho zkonstruuje
    // nekde v baliku rendering tedy bude nejaky katalog fraktalu a platforem


    @Override
    public void registerOutputTexture(OpenGLTexture output) {

    }

    @Override
    public void unregisterOutputTexture() {

    }

    @Override
    public void resize(int width, int height, OpenGLTexture output) {

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
}
