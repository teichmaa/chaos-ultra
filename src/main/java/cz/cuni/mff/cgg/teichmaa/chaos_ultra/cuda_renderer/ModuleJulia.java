package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DDoubleImmutable;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.cudaMemcpyKind;

import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.runtime.JCuda.cudaMemcpy;

public class ModuleJulia extends FractalRenderingModule {

    private static final String JULIA_C_PARAM_NAME = "julia_c";
    private static final long JULIA_C_PARAM_SIZE = 2 * Sizeof.DOUBLE;

    public ModuleJulia(Point2DDoubleImmutable c) {
        super("julia", "julia");

        //from https://github.com/jcuda/jcuda/issues/17
        juliaCParamPtr = new CUdeviceptr();
        long[] juliaCParamSizeArray = {0};
        cuModuleGetGlobal(juliaCParamPtr, juliaCParamSizeArray, getModule(), JULIA_C_PARAM_NAME);
        int juliaCParamSize = (int) juliaCParamSizeArray[0];
        if (juliaCParamSize != JULIA_C_PARAM_SIZE) {
            String message = "Wrong size of " + JULIA_C_PARAM_NAME + " constant in module " + getFractalName() + ". Expected: " + JULIA_C_PARAM_SIZE + "bytes, got: " + juliaCParamSize + "bytes";
            throw new CudaInitializationException(message);
        }

        setC(c);
    }

    private Point2DDoubleImmutable c;

    public Point2DDoubleImmutable getC() {
        return c;
    }

    public void setC(Point2DDoubleImmutable c) {
        this.c = c;

        double[] source = new double[]{c.getX(), c.getY()};
        cudaMemcpy(juliaCParamPtr, Pointer.to(source), JULIA_C_PARAM_SIZE, cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    private final CUdeviceptr juliaCParamPtr;
}
