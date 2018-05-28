package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

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
            throw new IllegalArgumentException("underSamplingLevel must be at least 1: " + underSamplingLevel);
        this.underSamplingLevel = underSamplingLevel;
        params[PARAM_IDX_UNDER_SAMPLING_LEVEL] = pointerTo(underSamplingLevel);
    }

    int getUnderSamplingLevel() {
        return underSamplingLevel;
    }

    @Override
    Pointer pointerToAbstractReal(double value) {
        return pointerTo((float)value);
    }

}
