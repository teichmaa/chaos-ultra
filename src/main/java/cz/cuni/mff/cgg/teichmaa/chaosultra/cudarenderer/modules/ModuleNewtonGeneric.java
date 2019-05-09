package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import com.google.gson.*;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.JsonHelpers;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDouble;

import java.util.List;
import java.util.stream.Collectors;

public class ModuleNewtonGeneric extends FractalRenderingModule {

    private static final String ROOTS_CONSTANT_NAME = "roots";
    private static final String COEFFS_CONSTANT_NAME = "coefficients";


    public ModuleNewtonGeneric() {
        super("newton_generic", "newton generic");
    }

    private List<Double> coefficients;
    private List<PointDouble> roots;

    @Override
    public void setFractalCustomParameters(String params) {
        parseJsonToParams(params);
        if(coefficients.size() != 4)
            throw new IllegalArgumentException("expecting 4 coefficients");
        if(roots.size() != 3)
            throw new IllegalArgumentException("expecting 3 roots");

        double[] rootsArr = new double[2 * 3];
        for (int i = 0; i < roots.size(); i++) {
            rootsArr[2 * i ] = roots.get(i).getX();
            rootsArr[2 * i + 1] = roots.get(i).getY();
        }
        writeToConstantMemory(ROOTS_CONSTANT_NAME, rootsArr);

        double[] coefsArr = new double[4];
        for (int i = 0; i < coefficients.size(); i++) {
            coefsArr[coefsArr.length - i -1] = coefficients.get(i); //the coefficient order is switched for the user and for the programmer
        }
        writeToConstantMemory(COEFFS_CONSTANT_NAME, coefsArr);
    }


    private void parseJsonToParams(String json){
        JsonObject jsonObject = JsonHelpers.parse(json);

        List<Double> coeffs = JsonHelpers.jsonArrayToList(jsonObject.get("coefficients").getAsJsonArray(), JsonElement::getAsDouble);
        List<List<Double>> roots = JsonHelpers.jsonArrayToList(jsonObject.get("roots").getAsJsonArray(), a -> JsonHelpers.jsonArrayToList(a.getAsJsonArray(), JsonElement::getAsDouble));

        if(roots.stream().map(List::size).anyMatch(size -> size != 2)){
            throw new IllegalArgumentException("Found a root that is not represented as [real, imag].");
        }

        this.coefficients = coeffs;
        this.roots = roots.stream().map(l -> PointDouble.of(l.get(0), l.get(1))).collect(Collectors.toList());
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(100);
        model.setFractalCustomParams("{ \"coefficients\" : [1, 0, 0, -1], \"roots\" : [ [1,0], [-0.5,0.86602540378] , [-0.5,-0.86602540378] ] }");
        //other nice value, is, for example:
       // model.setFractalCustomParams("{ \"coefficients\" : [1, 0, -2, 2], \"roots\" : [ [-1.7692923542386314,0], [0.884646177119315707620204,0.589742805022205501647280] , [0.884646177119315707,-0.589742805022205501] ] }");
    }
}
