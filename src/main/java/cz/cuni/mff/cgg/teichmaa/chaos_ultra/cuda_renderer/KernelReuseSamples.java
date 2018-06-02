package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;



import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelReuseSamples extends RenderingKernel {

    final short PARAM_IDX_INPUT;
    final short PARAM_IDX_INPUT_PITCH;
//    final short PARAM_IDX_OUTPUT;
//    final short PARAM_IDX_OUTPUT_PITCH;
//    final short PARAM_IDX_WIDTH;
//    final short PARAM_IDX_HEIGHT;
//    private final short PARAM_IDX_A;
//    private final short PARAM_IDX_B;
//    private final short PARAM_IDX_C;
//    private final short PARAM_IDX_D;
    private final short PARAM_IDX_P;
    private final short PARAM_IDX_Q;
    private final short PARAM_IDX_R;
    private final short PARAM_IDX_S;

    public KernelReuseSamples(CUmodule ownerModule) {
        super("fractalRenderReuseSamples", ownerModule);

        //PARAM_IDX_OUTPUT = registerParam();
//        PARAM_IDX_OUTPUT_PITCH = registerParam();
//        PARAM_IDX_WIDTH = registerParam();
//        PARAM_IDX_HEIGHT = registerParam();
//        PARAM_IDX_A = registerParam(0);
//        PARAM_IDX_B = registerParam(0);
//        PARAM_IDX_C = registerParam(0);
//        PARAM_IDX_D = registerParam(0);
        PARAM_IDX_P = registerParam(0);
        PARAM_IDX_Q = registerParam(0);
        PARAM_IDX_R = registerParam(0);
        PARAM_IDX_S = registerParam(0);
        PARAM_IDX_INPUT = registerParam();
        PARAM_IDX_INPUT_PITCH = registerParam();
    }

    @Override
    Pointer pointerToAbstractReal(double value) {
        return pointerTo((float) value);
    }

    public void setOriginBounds(double p, double q, double r, double s){
        params[PARAM_IDX_P] = pointerToAbstractReal(p);
        params[PARAM_IDX_Q] = pointerToAbstractReal(q);
        params[PARAM_IDX_R] = pointerToAbstractReal(r);
        params[PARAM_IDX_S] = pointerToAbstractReal(s);
    }
}
