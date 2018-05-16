package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;


import static cz.cuni.mff.cgg.teichmaa.mandelzoomer.view.RenderingModeFSM.RenderingMode.*;

/***
 * Finite State Machine for Rendering Mode transitions
 */
class RenderingModeFSM {

    enum RenderingMode {
        Waiting,
        ZoomingAuto,
        ZoomingOnce,
        Moving,
        ProgressiveRendering;
    }

    static final int MAX_PROGRESSIVE_RENDERING_LEVEL = 6;

    @Override
    public String toString() {
        String state = !zoomingAndMoving ? current.toString() : "ZoomingAndMoving";
        String lvl = isProgressiveRendering() ? " lvl " + PRlvl : "";
        return "FSM in state " + state + lvl;
    }

    private RenderingMode current = Waiting;
    private RenderingMode last = Waiting;
    private int PRlvl = 0;

    private boolean zoomingAndMoving = false;
    private boolean zoomingDirection = false;

    void reset() {
        current = Waiting;
        last = Waiting;
        zoomingAndMoving = false;
    }

    void step() {
        RenderingMode newValue = current;
        if ((current == Waiting && (last == ZoomingAuto || last == Moving))
                || current == ZoomingOnce
                ) {
            newValue = ProgressiveRendering;
            PRlvl = 0;
        } else if (current == ProgressiveRendering && PRlvl == MAX_PROGRESSIVE_RENDERING_LEVEL)
            newValue = Waiting;
        //default: do nothing
        last = current;
        current = newValue;
        if (current == ProgressiveRendering) {
            PRlvl = Math.min(MAX_PROGRESSIVE_RENDERING_LEVEL, PRlvl + 1);
        }
    }

    void doZoomingManualOnce(boolean inside) {
        last = current;
        current = ZoomingOnce;
        zoomingDirection = inside;
        zoomingAndMoving = false;
    }

    void startZoomingAndMoving(boolean inside) {
        startZooming(inside);
        zoomingAndMoving = true;

    }

    void startZooming(boolean inside) {
        last = current;
        current = ZoomingAuto;
        zoomingDirection = inside;
        zoomingAndMoving = false;
    }

    void stopZooming() {
        last = current;
        if (!zoomingAndMoving)
            current = Waiting;
        else
            current = Moving;
        zoomingAndMoving = false;
    }

    boolean isZooming() {
        return current == ZoomingAuto || zoomingAndMoving || current == ZoomingOnce;
    }

    boolean getZoomingDirection() {
        if (!isZooming()) throw new IllegalStateException("cannot ask for zooming direction when not zooming");
        return zoomingDirection;
    }

    void startMoving() {
        last = current;
        current = Moving;
    }

    boolean isMoving() {
        return current == Moving || zoomingAndMoving;
    }

    void stopMoving() {
        last = current;
        if (!zoomingAndMoving)
            current = Waiting;
        //else: current == Zooming{In,Out}, which is correct
        zoomingAndMoving = false;
    }

    void startProgressiveRendering() {
        last = current;
        current = ProgressiveRendering;
        PRlvl = 0;
        zoomingAndMoving = false;
    }

    int getProgressiveRenderingLevel() {
        if (!isProgressiveRendering())
            throw new IllegalStateException("cannot ask for Progressive rendering level when not Progressive rendering");
        return PRlvl;
    }

    boolean isProgressiveRendering() {
        return current == ProgressiveRendering;
    }

    boolean isWaiting() {
        return current == Waiting;
    }

    RenderingMode getCurrent() {
        return current;
    }

}
