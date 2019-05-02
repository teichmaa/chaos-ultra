package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params;


public interface SuperSampling {
    public int getSuperSamplingLevel();

    public void setSuperSamplingLevel(int superSamplingLevel);

    public boolean isUseAdaptiveSuperSampling();

    public void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling);

    public boolean isVisualiseSampleCount();

    public void setVisualiseSampleCount(boolean visualiseSampleCount);
}
