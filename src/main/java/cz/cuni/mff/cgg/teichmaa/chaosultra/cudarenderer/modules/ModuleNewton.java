package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;

public class ModuleNewton extends FractalRenderingModule {
    public ModuleNewton() {
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
