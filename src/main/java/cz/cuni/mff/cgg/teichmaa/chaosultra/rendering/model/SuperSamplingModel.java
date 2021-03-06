package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;


public interface SuperSamplingModel {
    float getMaxSuperSampling();

    void setMaxSuperSampling(float maxSuperSampling);

    boolean isUseAdaptiveSuperSampling();

    void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling);

    boolean isVisualiseSampleCount();

    void setVisualiseSampleCount(boolean visualiseSampleCount);
}
