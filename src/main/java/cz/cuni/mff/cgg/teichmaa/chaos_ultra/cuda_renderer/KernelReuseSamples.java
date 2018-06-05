package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;



import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelReuseSamples extends KernelMain {

    final short PARAM_IDX_INPUT;
    final short PARAM_IDX_INPUT_PITCH;
    private final short PARAM_IMAGE_REUSED;
    private final short PARAM_USE_FOVEATION;
    private final short PARAM_USE_SAMPLEREUSAL;

    public KernelReuseSamples(CUmodule ownerModule) {
        super("fractalRenderReuseSamples", ownerModule);

        PARAM_IMAGE_REUSED = registerParam(0);
        PARAM_IDX_INPUT = registerParam();
        PARAM_IDX_INPUT_PITCH = registerParam();
        PARAM_USE_FOVEATION = registerParam(1);
        PARAM_USE_SAMPLEREUSAL = registerParam(1);
    }

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo((float) value);
    }

    @Override
    Pointer pointerToAbstractReal(double v1, double v2, double v3, double v4) {
        return CudaHelpers.pointerTo((float) v1, (float) v2, (float) v3, (float) v4);
    }

    public void setOriginBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y){
        params[PARAM_IMAGE_REUSED] = pointerToAbstractReal(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }


    public void setUseFoveation(boolean value) {
        params[PARAM_USE_FOVEATION] = CudaHelpers.pointerTo(value);
    }

    public void setUseSampleReusal(boolean value) {
        params[PARAM_USE_SAMPLEREUSAL] = CudaHelpers.pointerTo(value);
    }
}
