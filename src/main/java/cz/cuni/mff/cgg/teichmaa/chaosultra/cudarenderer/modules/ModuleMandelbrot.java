package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;

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
