package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.RenderingModel;

public class ModuleMandelbrot extends FractalRenderingModule {


    public ModuleMandelbrot() {

        super("mandelbrot", "mandelbrot");
    }

    @Override
    public void setFractalCustomParameters(String params) {
        /* empty */
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setPlaneSegmentFromCenter(-0.5, 0, 2);
        model.setMaxIterations(800);
        model.setSuperSamplingLevel(5);
    }
}
