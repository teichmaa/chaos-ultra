package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.driver.CUmodule;

class KernelInit extends CudaKernel {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "init";

    public KernelInit(CUmodule ownerModule) {
        super(name, ownerModule);
    }

}
