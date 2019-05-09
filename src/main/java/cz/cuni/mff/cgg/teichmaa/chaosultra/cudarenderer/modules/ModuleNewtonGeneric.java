package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import com.google.gson.*;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.JsonHelpers;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDouble;

import java.util.List;
import java.util.Spliterator;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class ModuleNewtonGeneric extends FractalRenderingModule {
    public ModuleNewtonGeneric() {
        super("newton", "newton generic");
    }

    private double n;
    private List<Double> coefficients;
    private List<PointDouble> roots;

    @Override
    public void setFractalCustomParameters(String params) {
        JsonObject jsonObject = JsonHelpers.parse(params);

        int n = jsonObject.get("n").getAsInt();
        List<Double> coeffs = JsonHelpers.jsonArrayToList(jsonObject.get("coefficients").getAsJsonArray(), JsonElement::getAsDouble);
        List<List<Double>> roots = JsonHelpers.jsonArrayToList(jsonObject.get("roots").getAsJsonArray(), a -> JsonHelpers.jsonArrayToList(a.getAsJsonArray(), JsonElement::getAsDouble));
        if(n != roots.size()) {
            throw new IllegalArgumentException(String.format("N (%d) must match number of roots (%d).", n, roots.size()));
        } else if(n != coeffs.size() - 1 ){
            throw new IllegalArgumentException(String.format("N (%d) must eqal to number of coefficients - 1 (%d).", n, coeffs.size()));
        } else if(roots.stream().map(List::size).anyMatch(size -> size != 2)){
            throw new IllegalArgumentException("Found a root that is not represented as [real, imag].");
        }

        this.n = n;
        this.coefficients = coeffs;
        this.roots = roots.stream().map(l -> PointDouble.of(l.get(0), l.get(1))).collect(Collectors.toList());
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(100);
        model.setFractalCustomParams("{ \"n\" : 3, \"coefficients\" : [3, 0, 0, -1], \"roots\" : [ [1,0], [-0.5,0.86602540378] , [-0.5,-0.86602540378] ] }");
    }
}
