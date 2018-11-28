package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.IntegerProperty;

public interface SuperSampling {
    public int getSuperSamplingLevel();

    public IntegerProperty superSamplingLevelProperty();

    public void setSuperSamplingLevel(int superSamplingLevel);

    public boolean isUseAdaptiveSupersampling();

    public BooleanProperty useAdaptiveSupersamplingProperty();

    public void setUseAdaptiveSupersampling(boolean useAdaptiveSupersampling);

    public boolean isVisualiseSampleCount();

    public BooleanProperty visualiseSampleCountProperty();

    public void setVisualiseSampleCount(boolean visualiseSampleCount);
}
