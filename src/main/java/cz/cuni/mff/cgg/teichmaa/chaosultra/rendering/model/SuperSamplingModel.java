package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;


public interface SuperSamplingModel {
    int getSuperSamplingLevel();

    void setSuperSamplingLevel(int superSamplingLevel);

    boolean isUseAdaptiveSuperSampling();

    void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling);

    boolean isVisualiseSampleCount();

    void setVisualiseSampleCount(boolean visualiseSampleCount);
}
