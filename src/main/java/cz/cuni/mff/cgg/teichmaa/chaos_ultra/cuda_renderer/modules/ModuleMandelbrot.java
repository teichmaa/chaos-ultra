package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;

public class ModuleMandelbrot extends FractalRenderingModule {


    public ModuleMandelbrot() {

        super("mandelbrot", "mandelbrot");
    }

    @Override
    public void setFractalCustomParameters(String params) {
        /* empty */
    }
}
