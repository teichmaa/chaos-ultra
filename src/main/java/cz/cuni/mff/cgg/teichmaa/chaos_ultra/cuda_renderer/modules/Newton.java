package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.RenderingModel;

public class Newton extends FractalRenderingModule {
    public Newton() {
        super("newton", "newton");
    }

    @Override
    public void setFractalCustomParameters(String params) {

    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(100);
    }
}
