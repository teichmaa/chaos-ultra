package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelDebug extends CudaKernel {
    public KernelDebug(CUmodule ownerModule) {
        super("debug", ownerModule);

        short i1 = registerParam(-1L);
        short i2 = registerParam(20);
        short i3 = registerParam(30);

        params[i1] = Pointer.to(new int[]{15,16,17});
    }
}
