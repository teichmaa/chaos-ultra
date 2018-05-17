package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

class KernelMain extends RenderingKernel {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderMain";

    private final short PARAM_IDX_SUPER_SAMPLING_LEVEL;
    private final short PARAM_IDX_ADAPTIVE_SS;
    private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS;
    final short PARAM_IDX_RANDOM_SAMPLES;

    KernelMain(CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize params[] :
        PARAM_IDX_SUPER_SAMPLING_LEVEL = addParam(null);
        PARAM_IDX_ADAPTIVE_SS = addParam(null);
        PARAM_IDX_VISUALISE_ADAPTIVE_SS = addParam(null);
        PARAM_IDX_RANDOM_SAMPLES = addParam(null);

        setSuperSamplingLevel(1);
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
    }

    private int superSamplingLevel;
    private boolean adaptiveSS;
    private boolean visualiseAdaptiveSS;

    boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = Pointer.to(new int[]{visualiseAdaptiveSS ? 1 : 0});
    }

    boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = Pointer.to(new int[]{adaptiveSS ? 1 : 0});
    }


    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = Pointer.to(new int[]{superSamplingLevel});
    }

    int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

}
