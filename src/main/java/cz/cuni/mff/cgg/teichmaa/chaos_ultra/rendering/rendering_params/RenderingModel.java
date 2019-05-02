package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.FractalRenderer;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

/**
 * Represents all the rendering parameters that are used by current version of the Chaos Ultra project.
 */

public class RenderingModel implements FoveatedRendering, IterationLimit, SampleReuse, SuperSampling, DynamicFloatingPointPrecision, AutomaticQuality {

    public static final int SUPER_SAMPLING_MAX_LEVEL = FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

    private FloatPrecision floatingPointPrecision = FloatPrecision.defaultValue;
    private boolean useFoveatedRendering;
    private PointInt focus = new PointInt();
    private boolean zooming;
    private int maxIterations;
    private PlaneSegment segment = new PlaneSegment();
    private boolean useSampleReuse;
    private int superSamplingLevel;
    private boolean useAdaptiveSuperSampling;
    private boolean visualiseSampleCount;
    private boolean automaticQuality;
    private int canvasWidth;
    private int canvasHeight;
    private boolean sampleReuseCacheDirty;

    @Override
    public FloatPrecision getFloatingPointPrecision() {
        return floatingPointPrecision;
    }

    @Override
    public void setFloatingPointPrecision(FloatPrecision floatingPointPrecision) {
        this.floatingPointPrecision = floatingPointPrecision;
    }

    @Override
    public boolean isUseFoveatedRendering() {
        return useFoveatedRendering;
    }

    @Override
    public void setUseFoveatedRendering(boolean useFoveatedRendering) {
        this.useFoveatedRendering = useFoveatedRendering;
    }

    @Override
    public PointInt getFocus() {
        return focus;
    }

    @Override
    public void setFocus(PointInt focus) {
        this.focus = focus;
    }

    @Override
    public boolean isZooming() {
        return zooming;
    }

    @Override
    public void setZooming(boolean zooming) {
        this.zooming = zooming;
    }

    @Override
    public int getMaxIterations() {
        return maxIterations;
    }

    @Override
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public PlaneSegment getSegment() {
        return segment;
    }

    @Override
    public boolean isUseSampleReuse() {
        return useSampleReuse;
    }

    @Override
    public void setUseSampleReuse(boolean useSampleReuse) {
        this.useSampleReuse = useSampleReuse;
    }

    @Override
    public int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

    @Override
    /** Always clamps the value to be at least 1 and at most SUPER_SAMPLING_MAX_LEVEL
     */
    public void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = Math.max(1, Math.min(superSamplingLevel, SUPER_SAMPLING_MAX_LEVEL));
    }

    @Override
    public boolean isUseAdaptiveSuperSampling() {
        return useAdaptiveSuperSampling;
    }

    @Override
    public void setUseAdaptiveSuperSampling(boolean useAdaptiveSuperSampling) {
        this.useAdaptiveSuperSampling = useAdaptiveSuperSampling;
    }

    @Override
    public boolean isVisualiseSampleCount() {
        return visualiseSampleCount;
    }

    @Override
    public void setVisualiseSampleCount(boolean visualiseSampleCount) {
        this.visualiseSampleCount = visualiseSampleCount;
    }

    @Override
    public boolean isAutomaticQuality() {
        return automaticQuality;
    }

    @Override
    public void setAutomaticQuality(boolean automaticQuality) {
        this.automaticQuality = automaticQuality;
    }

    public int getCanvasWidth() {
        return canvasWidth;
    }

    public void setCanvasWidth(int canvasWidth) {
        this.canvasWidth = canvasWidth;
    }

    public int getCanvasHeight() {
        return canvasHeight;
    }

    public void setCanvasHeight(int canvasHeight) {
        this.canvasHeight = canvasHeight;
    }

    public boolean isSampleReuseCacheDirty() {
        return sampleReuseCacheDirty;
    }

    public void setSampleReuseCacheDirty(boolean sampleReuseCacheDirty) {
        this.sampleReuseCacheDirty = sampleReuseCacheDirty;
    }

    /**
     * @return deep copy of itself
     */
    public RenderingModel copy() {
        RenderingModel copy = new RenderingModel();
        copy.floatingPointPrecision = this.floatingPointPrecision;
        copy.useFoveatedRendering = this.useFoveatedRendering;
        copy.focus = this.focus.copy();
        copy.zooming = this.zooming;
        copy.maxIterations = this.maxIterations;
        copy.segment = this.segment.copy();
        copy.useSampleReuse = this.useSampleReuse;
        copy.superSamplingLevel = this.superSamplingLevel;
        copy.useAdaptiveSuperSampling = this.useAdaptiveSuperSampling;
        copy.visualiseSampleCount = this.visualiseSampleCount;
        copy.automaticQuality = this.automaticQuality;
        copy.canvasWidth = this.canvasWidth;
        copy.canvasHeight = this.canvasHeight;
        copy.sampleReuseCacheDirty = this.sampleReuseCacheDirty;
        return copy;
    }
}
