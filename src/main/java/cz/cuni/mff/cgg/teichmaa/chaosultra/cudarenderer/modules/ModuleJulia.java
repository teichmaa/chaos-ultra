package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDoubleImmutable;

public class ModuleJulia extends FractalRenderingModule {

    private static final String JULIA_C_PARAM_NAME = "julia_c";

    public ModuleJulia() {
        super("julia", "julia");
    }

    @Override
    public void initialize(){
        super.initialize();

        setC(PointDoubleImmutable.of(0,0));
    }

    private PointDoubleImmutable c;

    public PointDoubleImmutable getC() {
        return c;
    }

    public void setC(PointDoubleImmutable c) {
        this.c = c;
        writeToConstantMemory(JULIA_C_PARAM_NAME, c);
    }

    @Override
    public void setFractalCustomParameters(String params) {
        double[] vals = parseParamsAsDoubles(params);
        setC(PointDoubleImmutable.of(vals[0], vals[1]));
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setMaxIterations(400);
        model.setFractalCustomParams("-0.4;0.6");
        //other possible nice values:
        //model.setFractalCustomParams("-0.8;0.156");
        //model.setFractalCustomParams("0.285;0.01");
        //model.setFractalCustomParams("-1.77578;0");
    }
}
