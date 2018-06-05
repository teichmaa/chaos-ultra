package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

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
    }

}
