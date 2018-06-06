package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

class KernelAdvancedFloat extends KernelAdvanced {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderAdvancedFloat";

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo((float) value);
    }

    @Override
    Pointer pointerToAbstractReal(double v1, double v2, double v3, double v4) {
        return CudaHelpers.pointerTo((float) v1, (float) v2, (float) v3, (float) v4);
    }

    public KernelAdvancedFloat(CUmodule ownerModule) {
        super(name, ownerModule);
    }
}
