package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.NativePointerObject;
import jcuda.driver.CUmodule;

public class KernelCompose extends CudaKernel {

    final short PARAM_IDX_INPUT1 = 0;
    final short PARAM_IDX_INPUT1_PITCH = 1;
    final short PARAM_IDX_SURFACE_OUT = 2;
    final short PARAM_IDX_WIDTH = 3;
    final short PARAM_IDX_HEIGHT = 4;
    final short PARAM_IDX_SURFACE_PALETTE = 5;
    final short PARAM_IDX_PALETTE_LENGTH = 6;
    final short PARAM_IDX_DWELL = 7;

    public static final String name = "compose";

    public KernelCompose(CUmodule ownerModule) {
        super(name, ownerModule);
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        return new NativePointerObject[PARAM_IDX_DWELL+1];
    }
}
