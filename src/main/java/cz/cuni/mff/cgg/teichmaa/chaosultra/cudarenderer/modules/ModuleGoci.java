package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;

public class ModuleGoci extends FractalRenderingModule {


    public ModuleGoci() {
        super("goc", "goc");
    }

    @Override
    public void initialize() {
        super.initialize();
    }

    @Override
    public void setFractalCustomParameters(String params) {

    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(900);
    }
}
