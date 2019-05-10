package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;

public class ModuleTest extends FractalRenderingModule {


    public ModuleTest() {

        super("test", "test");
    }

    @Override
    public void setFractalCustomParameters(String params) {
        writeToConstantMemory("amplifier", Integer.parseInt(params));
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setFractalCustomParams("" + 10);
    }
}
