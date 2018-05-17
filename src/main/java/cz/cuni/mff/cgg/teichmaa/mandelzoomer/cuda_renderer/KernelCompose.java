package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.NativePointerObject;
import jcuda.driver.CUmodule;

public class KernelCompose extends CudaKernel {

    final short PARAM_IDX_INPUT_MAIN = 0;
    final short PARAM_IDX_INPUT_MAIN_PITCH = 1;
    final short PARAM_IDX_INPUT_BCG = 2;
    final short PARAM_IDX_INPUT_BCG_PITCH = 3;
    final short PARAM_IDX_SURFACE_OUT = 4;
    final short PARAM_IDX_WIDTH = 5;
    final short PARAM_IDX_HEIGHT = 6;
    final short PARAM_IDX_SURFACE_PALETTE = 7;
    final short PARAM_IDX_PALETTE_LENGTH = 8;
    final short PARAM_IDX_DWELL = 9;
    final short PARAM_IDX_MAIN_RADIUS = 10;
    final short PARAM_IDX_FOCUS_X = 11;
    final short PARAM_IDX_FOCUS_Y = 12;

    public static final String name = "compose";

    public KernelCompose(CUmodule ownerModule) {
        super(name, ownerModule);
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        return new NativePointerObject[PARAM_IDX_FOCUS_Y+1];
    }
}
