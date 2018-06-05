package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.driver.CUmodule;

abstract class KernelMain extends RenderingKernel {

    // uint maxSuperSampling,
    // Pointi focus, bool adaptiveSS, bool visualiseSS,


    private final short PARAM_IDX_MAX_SUPER_SAMPLING;
    private final short PARAM_IDX_ADAPTIVE_SS;
    private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS;

    KernelMain(String name, CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize params[] :
        PARAM_IDX_MAX_SUPER_SAMPLING = registerParam(1);
        PARAM_IDX_ADAPTIVE_SS = registerParam(1);
        PARAM_IDX_VISUALISE_ADAPTIVE_SS = registerParam(0);

        setSuperSamplingLevel(1);
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
    }

    private int superSamplingLevel;
    private boolean adaptiveSS;
    private boolean visualiseAdaptiveSS;

    boolean getVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = CudaHelpers.pointerTo(visualiseAdaptiveSS);
    }

    boolean getAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = CudaHelpers.pointerTo(adaptiveSS);
    }

    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_MAX_SUPER_SAMPLING] = CudaHelpers.pointerTo(superSamplingLevel);
    }

    int getSuperSamplingLevel() {
        return superSamplingLevel;
    }


}
