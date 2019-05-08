package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

/**
 * The data needed by a {@code FractalRenderer} to render and methods to give feedback
 */
public interface RenderingModel extends FoveatedRenderingModel, IterationLimitModel, SampleReuseModel, SuperSamplingModel, DynamicFloatingPointPrecisionModel, AutomaticQualityModel, PublicErrorLogger {

    PlaneSegment getPlaneSegment();

    RenderingModel copy();

}
