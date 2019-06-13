package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

public interface RenderingStateModel {

    /**
     * @return true for zooming in, false for zooming out
     */
    boolean getZoomingDirection() ;

    boolean isProgressiveRendering() ;

    boolean isWaiting() ;

    boolean wasProgressiveRendering() ;

    boolean isMoving();

    int getProgressiveRenderingLevel();

    boolean isZooming();

    void resetState();

    boolean isDifferentThanLast();

    RenderingModeFSM.RenderingMode getCurrent();
}
