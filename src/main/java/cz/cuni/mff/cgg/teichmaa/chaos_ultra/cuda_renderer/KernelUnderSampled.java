package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

class KernelUnderSampled extends RenderingKernel {
    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderUnderSampled";

    private final short PARAM_IDX_UNDER_SAMPLING_LEVEL;

    KernelUnderSampled(CUmodule ownerModule) {
        super(name, ownerModule);
        PARAM_IDX_UNDER_SAMPLING_LEVEL = registerParam();

        setUnderSamplingLevel(1);
    }

    private int underSamplingLevel;

    void setUnderSamplingLevel(int underSamplingLevel) {
        if(underSamplingLevel < 1)
            throw new IllegalArgumentException("underSamplingLevel must be at least 1, current value: " + underSamplingLevel);
        this.underSamplingLevel = underSamplingLevel;
        params[PARAM_IDX_UNDER_SAMPLING_LEVEL] = CudaHelpers.pointerTo(underSamplingLevel);
    }

    int getUnderSamplingLevel() {
        return underSamplingLevel;
    }

    @Override
    Pointer pointerToAbstractReal(double value) {
        return CudaHelpers.pointerTo((float)value);
    }

    @Override
    Pointer pointerToAbstractReal(double v1, double v2, double v3, double v4) {
        return CudaHelpers.pointerTo((float) v1, (float) v2, (float) v3, (float) v4);
    }
}
