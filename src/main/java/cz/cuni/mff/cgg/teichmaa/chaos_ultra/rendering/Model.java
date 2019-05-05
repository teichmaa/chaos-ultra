package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui.GUIModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.PlaneSegment;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.RenderingModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

import java.util.ArrayList;
import java.util.Collection;

import static cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

/**
 * Represents all the rendering parameters that are used by current version of the Chaos Ultra project.
 */

class Model implements RenderingModel, GUIModel, DefaultFractalModel {

    private FloatPrecision floatingPointPrecision = FloatPrecision.defaultValue;
    private boolean useFoveatedRendering;
    private PointInt mouseFocus = new PointInt();
    private boolean zooming;
    private int maxIterations;
    private PlaneSegment planeSegment = new PlaneSegment();
    private boolean useSampleReuse;
    private int superSamplingLevel;
    private boolean useAdaptiveSuperSampling;
    private boolean visualiseSampleCount;
    private boolean automaticQuality;
    private int canvasWidth;
    private int canvasHeight;
    private boolean sampleReuseCacheDirty;
    private PointInt lastMousePosition = new PointInt();
    private Collection<String> availableFractals;
    private String fractalName;
    private String fractalCustomParams;

    /**
     * @return deep copy of itself
     */
    public Model copy() {
        Model copy = new Model();
        copy.floatingPointPrecision = this.floatingPointPrecision;
        copy.useFoveatedRendering = this.useFoveatedRendering;
        copy.mouseFocus = this.mouseFocus.copy();
        copy.zooming = this.zooming;
        copy.maxIterations = this.maxIterations;
        copy.planeSegment = this.planeSegment.copy();
        copy.useSampleReuse = this.useSampleReuse;
        copy.superSamplingLevel = this.superSamplingLevel;
        copy.useAdaptiveSuperSampling = this.useAdaptiveSuperSampling;
        copy.visualiseSampleCount = this.visualiseSampleCount;
        copy.automaticQuality = this.automaticQuality;
        copy.canvasWidth = this.canvasWidth;
        copy.canvasHeight = this.canvasHeight;
        copy.sampleReuseCacheDirty = this.sampleReuseCacheDirty;
        copy.lastMousePosition = this.lastMousePosition;
        copy.availableFractals = new ArrayList<>(availableFractals);
        copy.fractalName = this.fractalName;
        copy.fractalCustomParams = this.fractalCustomParams;
        return copy;
    }

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
    public PointInt getMouseFocus() {
        return mouseFocus;
    }

    @Override
    public void setMouseFocus(PointInt mouseFocus) {
        this.mouseFocus = mouseFocus;
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

    public PlaneSegment getPlaneSegment() {
        return planeSegment;
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
        setSampleReuseCacheDirty(true);
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

    void setCanvasWidth(int canvasWidth) {
        this.canvasWidth = canvasWidth;
    }

    public int getCanvasHeight() {
        return canvasHeight;
    }

    void setCanvasHeight(int canvasHeight) {
        this.canvasHeight = canvasHeight;
    }

    public boolean isSampleReuseCacheDirty() {
        return sampleReuseCacheDirty;
    }

    public void setSampleReuseCacheDirty(boolean sampleReuseCacheDirty) {
        this.sampleReuseCacheDirty = sampleReuseCacheDirty;
    }

    PointInt getLastMousePosition() {
        return lastMousePosition;
    }

    @Override
    public Collection<String> getAvailableFractals() {
        return availableFractals;
    }

    public void setAvailableFractals(Collection<String> availableFractals) {
        this.availableFractals = availableFractals;
    }

    public String getFractalName() {
        return fractalName;
    }

    public void setFractalName(String fractalName) {
        this.fractalName = fractalName;
    }

    @Override
    public void setPlaneSegmentFromCenter(double centerX, double centerY, double zoom) {
        double windowRelHeight = 1;
        double windowRelWidth = windowRelHeight / (double) canvasHeight * canvasWidth;
        double segment_left_bottom_x = centerX - windowRelWidth * zoom / 2;
        double segment_left_bottom_y = centerY - windowRelHeight * zoom / 2;
        double segment_right_top_x = centerX + windowRelWidth * zoom / 2;
        double segment_right_top_y = centerY + windowRelHeight * zoom / 2;

        this.planeSegment.setAll(segment_left_bottom_x, segment_left_bottom_y, segment_right_top_x, segment_right_top_y);
    }

    public String getFractalCustomParams() {
        return fractalCustomParams;
    }

    @Override
    public void setFractalCustomParams(String fractalCustomParams) {
        this.fractalCustomParams = fractalCustomParams;
    }
}