package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.RenderingModel;
import jcuda.driver.CUmodule;

public class KernelCompose extends CudaKernel {

    final short PARAM_IDX_INPUT_MAIN;
    final short PARAM_IDX_INPUT_MAIN_PITCH;
    final short PARAM_IDX_INPUT_BCG;
    final short PARAM_IDX_INPUT_BCG_PITCH;
    final short PARAM_IDX_SURFACE_OUT;
    final short PARAM_IDX_WIDTH;
    final short PARAM_IDX_HEIGHT;
    final short PARAM_IDX_SURFACE_PALETTE;
    final short PARAM_IDX_PALETTE_LENGTH;
    final short PARAM_IDX_MAX_SUPER_SAMPLING;

    public static final String name = "compose";

    public KernelCompose(CUmodule ownerModule) {
        super(name, ownerModule);

        PARAM_IDX_INPUT_MAIN = registerParam();
        PARAM_IDX_INPUT_MAIN_PITCH = registerParam();
        PARAM_IDX_INPUT_BCG = registerParam();
        PARAM_IDX_INPUT_BCG_PITCH = registerParam();
        PARAM_IDX_SURFACE_OUT = registerParam();
        PARAM_IDX_WIDTH = registerParam();
        PARAM_IDX_HEIGHT = registerParam();
        PARAM_IDX_SURFACE_PALETTE = registerParam();
        PARAM_IDX_PALETTE_LENGTH = registerParam();
        PARAM_IDX_MAX_SUPER_SAMPLING = registerParam();
    }

    @Override
    public void setParamsFromModel(RenderingModel model) {
        super.setParamsFromModel(model);
        getKernelParams()[PARAM_IDX_MAX_SUPER_SAMPLING] = CudaHelpers.pointerTo(model.getMaxSuperSampling());
    }
}
