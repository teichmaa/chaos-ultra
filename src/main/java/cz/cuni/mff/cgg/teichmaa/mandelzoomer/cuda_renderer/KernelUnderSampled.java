package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.NativePointerObject;
import jcuda.driver.CUmodule;

public class KernelUnderSampled extends RenderingKernel {
    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderUnderSampled";

    public KernelUnderSampled(CUmodule ownerModule) {
        super(name, ownerModule);
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        return new NativePointerObject[0];
    }
}
