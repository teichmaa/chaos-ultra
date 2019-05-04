package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

public interface RenderingModel extends FoveatedRendering, IterationLimit, SampleReuse, SuperSampling, DynamicFloatingPointPrecision, AutomaticQuality {

    PlaneSegment getPlaneSegment();

    void setSampleReuseCacheDirty(boolean b);

    boolean isSampleReuseCacheDirty();

    RenderingModel copy();

}
