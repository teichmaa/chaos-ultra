package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer.CudaKernel;
import jcuda.NativePointerObject;
import jcuda.driver.CUmodule;

class KernelInit extends CudaKernel {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "init";

    public KernelInit(CUmodule ownerModule) {
        super(name, ownerModule);
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        return new NativePointerObject[0];
    }
}
