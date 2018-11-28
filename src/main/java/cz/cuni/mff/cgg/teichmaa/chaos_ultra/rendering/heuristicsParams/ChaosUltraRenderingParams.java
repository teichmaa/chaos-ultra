package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import javafx.beans.property.*;

/**
 * Represents all the rendering parameters that are used by current version of the Chaos Ultra project.
 */

public class ChaosUltraRenderingParams implements FoveatedRendering, IterationLimit, SampleReuse, SuperSampling, DynamicFloatingPointPrecision {

    private IntegerProperty maxIterations = new SimpleIntegerProperty();

    @Override
    public int getMaxIterations() {
        return maxIterations.get();
    }

    @Override
    public IntegerProperty maxIterationsProperty() {
        return maxIterations;
    }

    @Override
    public void setMaxIterations(int maxIterations) {
        this.maxIterations.set(maxIterations);
    }


    private BooleanProperty useFoveatedRendering = new SimpleBooleanProperty();

    @Override
    public boolean isUseFoveatedRendering() {
        return useFoveatedRendering.get();
    }

    @Override
    public BooleanProperty useFoveatedRenderingProperty() {
        return useFoveatedRendering;
    }

    @Override
    public void setUseFoveatedRendering(boolean useFoveatedRendering) {
        this.useFoveatedRendering.set(useFoveatedRendering);
    }


    private BooleanProperty useSampleReuse = new SimpleBooleanProperty();

    @Override
    public boolean isUseSampleReuse() {
        return useSampleReuse.get();
    }

    @Override
    public BooleanProperty useSampleReuseProperty() {
        return useSampleReuse;
    }

    @Override
    public void setUseSampleReuse(boolean useSampleReuse) {
        this.useSampleReuse.set(useSampleReuse);
    }


    private IntegerProperty superSamplingLevel = new SimpleIntegerProperty();
    private BooleanProperty useAdaptiveSupersampling = new SimpleBooleanProperty();
    private BooleanProperty visualiseSampleCount = new SimpleBooleanProperty();

    @Override
    public int getSuperSamplingLevel() {
        return superSamplingLevel.get();
    }

    @Override
    public IntegerProperty superSamplingLevelProperty() {
        return superSamplingLevel;
    }

    @Override
    public void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel.set(superSamplingLevel);
    }

    @Override
    public boolean isUseAdaptiveSupersampling() {
        return useAdaptiveSupersampling.get();
    }

    @Override
    public BooleanProperty useAdaptiveSupersamplingProperty() {
        return useAdaptiveSupersampling;
    }

    @Override
    public void setUseAdaptiveSupersampling(boolean useAdaptiveSupersampling) {
        this.useAdaptiveSupersampling.set(useAdaptiveSupersampling);
    }

    @Override
    public boolean isVisualiseSampleCount() {
        return visualiseSampleCount.get();
    }

    @Override
    public BooleanProperty visualiseSampleCountProperty() {
        return visualiseSampleCount;
    }

    @Override
    public void setVisualiseSampleCount(boolean visualiseSampleCount) {
        this.visualiseSampleCount.set(visualiseSampleCount);
    }


    private ObjectProperty<FloatPrecision> floatingPointPrecision = new SimpleObjectProperty<>();

    @Override
    public FloatPrecision getFloatingPointPrecision() {
        return floatingPointPrecision.get();
    }

    @Override
    public ObjectProperty<FloatPrecision> floatingPointPrecisionProperty() {
        return floatingPointPrecision;
    }

    @Override
    public void setFloatingPointPrecision(FloatPrecision floatingPointPrecision) {
        this.floatingPointPrecision.set(floatingPointPrecision);
    }

    private BooleanProperty automaticQuality = new SimpleBooleanProperty();

    public boolean isAutomaticQuality() {
        return automaticQuality.get();
    }

    public BooleanProperty automaticQualityProperty() {
        return automaticQuality;
    }

    public void setAutomaticQuality(boolean automaticQuality) {
        this.automaticQuality.set(automaticQuality);
    }

}
