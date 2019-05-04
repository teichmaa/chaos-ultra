package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

public interface DefaultFractalModel extends RenderingModel {

    void setPlaneSegmentFromCenter(double centerX, double centerY, double zoom);

    void setFractalCustomParams(String text);
}
