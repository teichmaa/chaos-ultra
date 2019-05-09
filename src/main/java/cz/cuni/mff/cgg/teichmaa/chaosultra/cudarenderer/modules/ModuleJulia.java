package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.CuSizedDeviceptr;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.CudaInitializationException;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDouble;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDoubleImmutable;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDoubleReadable;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.runtime.JCuda.cudaMemcpy;

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
        model.setFractalCustomParams("-0.4;0.6");
    }
}
