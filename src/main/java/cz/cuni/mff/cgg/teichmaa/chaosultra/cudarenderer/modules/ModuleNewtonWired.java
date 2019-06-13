package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;

public class ModuleNewtonWired extends FractalRenderingModule {
    public ModuleNewtonWired() {
        super("newton_wired", "newton wired");
    }

    @Override
    public void setFractalCustomParameters(String params) {

    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(200);
    }
}
