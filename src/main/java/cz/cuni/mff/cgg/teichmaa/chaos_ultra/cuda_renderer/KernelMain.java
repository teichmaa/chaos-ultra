package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.driver.CUmodule;

abstract class KernelMain extends RenderingKernel {

    // uint maxSuperSampling,
    // Pointi focus, bool adaptiveSS, bool visualiseSS,


    private final short PARAM_IDX_MAX_SUPER_SAMPLING;
    protected final short PARAM_IDX_FLAGS;

    private final static int USE_ADAPTIVE_SS_FLAG_IDX = 0;

    KernelMain(String name, CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize params[] :
        PARAM_IDX_MAX_SUPER_SAMPLING = registerParam(1);
        PARAM_IDX_FLAGS = registerParam(1);

        setSuperSamplingLevel(1);
        setAdaptiveSS(true);
    }

    private int superSamplingLevel;
    private boolean adaptiveSS;
    protected BitMask flags = new BitMask();

    boolean getAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        flags.setBit(USE_ADAPTIVE_SS_FLAG_IDX, adaptiveSS);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }

    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_MAX_SUPER_SAMPLING] = CudaHelpers.pointerTo(superSamplingLevel);
    }

    int getSuperSamplingLevel() {
        return superSamplingLevel;
    }


}
