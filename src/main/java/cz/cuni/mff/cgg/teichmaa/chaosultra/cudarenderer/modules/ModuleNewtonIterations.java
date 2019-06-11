package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import com.google.gson.JsonObject;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.JsonHelpers;

public class ModuleNewtonIterations extends ModuleNewtonGeneric {

    public static final String COLOR_MAGNIFIER_PARAM_NAME = "colorMagnifier";

    public ModuleNewtonIterations() {
        super("newton_iterations", "newton colored by iterations");
    }

    @Override
    public void setFractalCustomParameters(String params) {
        super.setFractalCustomParameters(params);

        JsonObject json = JsonHelpers.parse(params);
        if(json.has(COLOR_MAGNIFIER_PARAM_NAME)){
            int colorMagnifier = json.get(COLOR_MAGNIFIER_PARAM_NAME).getAsInt();
            writeToConstantMemory(COLOR_MAGNIFIER_PARAM_NAME, colorMagnifier);
        }
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        String params = "{\"" + COLOR_MAGNIFIER_PARAM_NAME + "\": 11,"
                + model.getFractalCustomParams().substring(1);

        model.setFractalCustomParams(params);
    }
}
