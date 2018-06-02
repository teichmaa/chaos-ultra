package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelMainFloat extends KernelMain {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderMainFloat";

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo((float) value);
    }

    public KernelMainFloat(CUmodule ownerModule) {
        super(name, ownerModule);
    }
}
