package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;


import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.PlaneSegment;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.RenderingModel;
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

    @Override
    public void setParamsFromModel(RenderingModel model) {
        super.setParamsFromModel(model);
        setFocus(model.getMouseFocus().getX(), model.getMouseFocus().getY());
        setIsZooming(model.isZooming());
        setUseFoveation(model.isUseFoveatedRendering());
        setUseSampleReuse(model.isUseSampleReuse());
    }

    public void setOriginSegment(PlaneSegment segment){
        setOriginSegment(
                segment.getLeftBottom().getX(),
                segment.getLeftBottom().getY(),
                segment.getRightTop().getX(),
                segment.getRightTop().getY()
        );
    }

    void setOriginSegment(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y){
        checkArgument(left_bottom_x, "origin segment left_bottom_x");
        checkArgument(left_bottom_y, "origin segment left_bottom_y");
        checkArgument(right_top_x, "origin segment right_top_x");
        checkArgument(right_top_y, "origin segment right_top_y");
        params[PARAM_IDX_IMAGE_REUSED] = pointerToAbstractReal(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

    void setInput(CUdeviceptr input, long inputPitch){
        params[PARAM_IDX_INPUT] = Pointer.to(input);
        params[PARAM_IDX_INPUT_PITCH] = CudaHelpers.pointerTo(inputPitch);
    }

    public void setFocus(int x, int y){
        params[PARAM_IDX_FOCUS] = CudaHelpers.pointerTo(x, y);
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
