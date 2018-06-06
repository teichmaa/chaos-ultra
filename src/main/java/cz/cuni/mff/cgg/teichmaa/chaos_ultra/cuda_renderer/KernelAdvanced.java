package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;



import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmodule;

abstract class KernelAdvanced extends KernelMain {

    private final short PARAM_IDX_INPUT;
    private final short PARAM_IDX_INPUT_PITCH;
    private final short PARAM_IDX_IMAGE_REUSED;
    private final short PARAM_IDX_FOCUS;

    private final static int USE_FOVEATION_FLAG_IDX = 2;
    private final static int USE_SAMPLE_REUSE_FLAG_IDX = 3;
    private final static int IS_ZOOMING_FLAG_IDX = 4;


    public KernelAdvanced(String name, CUmodule ownerModule) {
        super(name, ownerModule);

        PARAM_IDX_IMAGE_REUSED = registerParam(0);
        PARAM_IDX_INPUT = registerParam();
        PARAM_IDX_INPUT_PITCH = registerParam();
        PARAM_IDX_FOCUS = registerParam(0);
    }

    private int focus_x;
    private int focus_y;

    public void setOriginBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y){
        params[PARAM_IDX_IMAGE_REUSED] = pointerToAbstractReal(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

    void setInput(CUdeviceptr input, long inputPitch){
        params[PARAM_IDX_INPUT] = Pointer.to(input);
        params[PARAM_IDX_INPUT_PITCH] = CudaHelpers.pointerTo(inputPitch);
    }

    public void setFocus(int x, int y){
        params[PARAM_IDX_FOCUS] = CudaHelpers.pointerTo(x, y);
        this.focus_x = x;
        this.focus_y = y;
    }

    public int getFocusX() {
        return focus_x;
    }

    public int getFocusY() {
        return focus_y;
    }

    public void setIsZooming(boolean zooming){
        flags.setBit(IS_ZOOMING_FLAG_IDX, zooming);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }


    public void setUseFoveation(boolean value) {
        flags.setBit(USE_FOVEATION_FLAG_IDX, value);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }

    public void setUseSampleReuse(boolean value) {
        flags.setBit(USE_SAMPLE_REUSE_FLAG_IDX, value);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }
}
