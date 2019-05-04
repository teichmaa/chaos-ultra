package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

public interface RenderingStateModel {

    boolean getZoomingDirection() ;

    boolean isProgressiveRendering() ;

    boolean isWaiting() ;

    boolean wasProgressiveRendering() ;

    boolean isMoving();

    int getProgressiveRenderingLevel();

    boolean isZooming();

    void step();

    void resetState();
}
