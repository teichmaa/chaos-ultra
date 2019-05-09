package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

/**
 * Allows fractals to supply default values
 */
public interface DefaultFractalModel extends RenderingModel {

    void setPlaneSegmentFromCenter(double centerX, double centerY, double zoom);

    void setFractalCustomParams(String text);
}
