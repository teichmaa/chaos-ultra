package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.CudaInitializationException;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDoubleImmutable;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.runtime.JCuda.cudaMemcpy;

public class ModuleJulia extends FractalRenderingModule {

    private static final String JULIA_C_PARAM_NAME = "julia_c";
    private static final long JULIA_C_PARAM_SIZE = 2 * Sizeof.DOUBLE;

    public ModuleJulia() {
        super("julia", "julia");
    }

    @Override
    public void initialize(){
        super.initialize();

        //initialize juliaCParamSizeArray to point to device memory containing the Julia-C-parameter
        //from https://github.com/jcuda/jcuda/issues/17
        long[] juliaCParamSizeArray = {0};
        cuModuleGetGlobal(juliaCParamPtr, juliaCParamSizeArray, getModule(), JULIA_C_PARAM_NAME);
        int juliaCParamSize = (int) juliaCParamSizeArray[0];
        if (juliaCParamSize != JULIA_C_PARAM_SIZE) {
            String message = "Wrong size of " + JULIA_C_PARAM_NAME + " constant in module " + getFractalName() + ". Expected: " + JULIA_C_PARAM_SIZE + "bytes, got: " + juliaCParamSize + "bytes";
            throw new CudaInitializationException(message);
        }

        setC(PointDoubleImmutable.of(0,0));
    }

    private final CUdeviceptr juliaCParamPtr = new CUdeviceptr();
    private PointDoubleImmutable c;

    public PointDoubleImmutable getC() {
        return c;
    }

    public void setC(PointDoubleImmutable c) {
        this.c = c;

        double[] source = new double[]{c.getX(), c.getY()};
        cudaMemcpy(juliaCParamPtr, Pointer.to(source), JULIA_C_PARAM_SIZE, cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    @Override
    public void setFractalCustomParameters(String params) {
        double[] vals = parseParamsAsDoubles(params);
        setC(PointDoubleImmutable.of(vals[0], vals[1]));
    }

    @Override
    protected void supplyDefaultValues(DefaultFractalModel model) {
        super.supplyDefaultValues(model);
        model.setFractalCustomParams("0.3;0.3");
    }
}
