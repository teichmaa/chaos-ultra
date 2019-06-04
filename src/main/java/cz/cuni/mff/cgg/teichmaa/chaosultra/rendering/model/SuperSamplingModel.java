package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;


public interface SuperSamplingModel {
    int getMaxSuperSampling();

    void setMaxSuperSampling(int maxSuperSampling);

    boolean isUseAdaptiveSuperSampling();

    void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling);

    boolean isVisualiseSampleCount();

    void setVisualiseSampleCount(boolean visualiseSampleCount);
}
