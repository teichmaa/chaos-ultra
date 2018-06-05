package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

public class KernelMainDouble extends KernelMain {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderMainDouble";

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo(value);
    }

    @Override
    Pointer pointerToAbstractReal(double v1, double v2, double v3, double v4) {
        return CudaHelpers.pointerTo(v1, v2, v3, v4);
    }

    public KernelMainDouble(CUmodule ownerModule) {
        super(name, ownerModule);
    }
}
