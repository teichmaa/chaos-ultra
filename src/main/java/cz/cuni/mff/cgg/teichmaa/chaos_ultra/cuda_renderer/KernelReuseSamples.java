package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;



import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelReuseSamples extends KernelMain {

    final short PARAM_IDX_INPUT;
    final short PARAM_IDX_INPUT_PITCH;
    private final short PARAM_IDX_P;
    private final short PARAM_IDX_Q;
    private final short PARAM_IDX_R;
    private final short PARAM_IDX_S;

    public KernelReuseSamples(CUmodule ownerModule) {
        super("fractalRenderReuseSamples", ownerModule);

        PARAM_IDX_P = registerParam(0);
        PARAM_IDX_Q = registerParam(0);
        PARAM_IDX_R = registerParam(0);
        PARAM_IDX_S = registerParam(0);
        PARAM_IDX_INPUT = registerParam();
        PARAM_IDX_INPUT_PITCH = registerParam();
    }

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo((float) value);
    }

    public void setOriginBounds(double p, double q, double r, double s){
        params[PARAM_IDX_P] = pointerToAbstractReal(p);
        params[PARAM_IDX_Q] = pointerToAbstractReal(q);
        params[PARAM_IDX_R] = pointerToAbstractReal(r);
        params[PARAM_IDX_S] = pointerToAbstractReal(s);
    }


}
