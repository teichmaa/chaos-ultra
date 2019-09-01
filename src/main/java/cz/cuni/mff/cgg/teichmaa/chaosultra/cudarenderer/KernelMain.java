package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.RenderingModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.BitField;
import jcuda.driver.CUmodule;

abstract class KernelMain extends RenderingKernel {

    private final short PARAM_IDX_MAX_SUPER_SAMPLING;
    protected final short PARAM_IDX_FLAGS;

    private final static int USE_ADAPTIVE_SS_FLAG_IDX = 0;
    private final static int VISUALISE_SAMPLE_COUNT_FLAG_IDX = 1;

    KernelMain(String name, CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize heuristicsParams[] :
        PARAM_IDX_MAX_SUPER_SAMPLING = registerParam(1);
        PARAM_IDX_FLAGS = registerParam(1);

        setMaxSuperSampling(1);
        setUseAdaptiveSuperSampling(true);
    }

    @Override
    public void setParamsFromModel(RenderingModel model) {
        super.setParamsFromModel(model);
        setVisualiseSampleCount(model.isVisualiseSampleCount());
        setUseAdaptiveSuperSampling(model.isUseAdaptiveSuperSampling());
        setMaxSuperSampling(model.getMaxSuperSampling());
    }

    private float maxSuperSampling;
    private boolean useAdaptiveSuperSampling;
    protected BitField flags = new BitField();

    boolean isVisualiseSampleCount(){
        return flags.getBit(VISUALISE_SAMPLE_COUNT_FLAG_IDX);
    }

    void setVisualiseSampleCount(boolean visualise) {
        flags.setBit(VISUALISE_SAMPLE_COUNT_FLAG_IDX, visualise);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }

    boolean getUseAdaptiveSuperSampling() {
        return useAdaptiveSuperSampling;
    }

    void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling) {
        this.useAdaptiveSuperSampling = useAdaptiveSuperSampling;
        flags.setBit(USE_ADAPTIVE_SS_FLAG_IDX, useAdaptiveSuperSampling);
        params[PARAM_IDX_FLAGS] = CudaHelpers.pointerTo(flags.getValue());
    }

    void setMaxSuperSampling(float maxSuperSampling) {
        this.maxSuperSampling = maxSuperSampling;
        params[PARAM_IDX_MAX_SUPER_SAMPLING] = CudaHelpers.pointerTo(maxSuperSampling);
    }

    float getMaxSuperSampling() {
        return maxSuperSampling;
    }

}
