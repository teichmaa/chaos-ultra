package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

/**
 * The data needed by a {@code FractalRenderer} to render and methods to give feedback
 */
public interface RenderingModel extends FoveatedRenderingModel, IterationLimitModel, SampleReuseModel, SuperSamplingModel, DynamicFloatingPointPrecisionModel, AutomaticQualityModel, PublicErrorLogger {

    PlaneSegment getPlaneSegment();

    RenderingModel copy();

    boolean isZooming();

    /**
     * If isZooming, whether zooming in or out. If !isZooming, the value is undefined.
     *
     * @return true for zoom in, false for zoom out, undefined value if not zooming.
     */
    boolean isZoomingIn();
}
