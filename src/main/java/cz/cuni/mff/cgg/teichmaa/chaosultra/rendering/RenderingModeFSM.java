package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;


import static cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.RenderingModeFSM.RenderingMode.*;

/***
 * Finite State Machine for Rendering Mode transitions
 */
class RenderingModeFSM implements RenderingStateModel {

    enum RenderingMode {
        Waiting,
        ZoomingAuto,
        ZoomingOnce,
        Moving,
        ProgressiveRendering
    }

    static final int MAX_PROGRESSIVE_RENDERING_LEVEL = 6;

    @Override
    public String toString() {
        String state = !zoomingAndMoving ? current.toString() : "ZoomingAndMoving";
        String lvl = isProgressiveRendering() ? " lvl " + PRLvl : "";
        return "FSM in state " + state + lvl;
    }

    private RenderingMode current = Waiting;
    private RenderingMode last = Waiting;
    /**
     * Progressive rendering level
     */
    private int PRLvl = 0;

    private boolean zoomingAndMoving = false;
    /**
     * true for inside, false for outside
     */
    private boolean zoomingDirection = false;

    public void resetState() {
        last = current;
        current = Waiting;
        zoomingAndMoving = false;
    }

    public void step() {
        RenderingMode newValue;
        if ((current == Waiting && (last == ZoomingAuto || last == Moving))
                || current == ZoomingOnce
        ) {
            newValue = ProgressiveRendering;
            PRLvl = -1; //will be increased at the end of the method
        } else if (current == ProgressiveRendering && PRLvl >= MAX_PROGRESSIVE_RENDERING_LEVEL) {
            newValue = Waiting;
        } else {
            newValue = current; //default: do not change the state
        }
        last = current;
        current = newValue;

        if (current == ProgressiveRendering) {
            PRLvl = Math.min(MAX_PROGRESSIVE_RENDERING_LEVEL, PRLvl + 1);
        }
    }

    public void doZoomingManualOnce(boolean inside) {
        last = current;
        current = ZoomingOnce;
        zoomingDirection = inside;
        zoomingAndMoving = false;
    }

    public void startZoomingAndMoving(boolean inside) {
        startZooming(inside);
        zoomingAndMoving = true;

    }

    public void startZooming(boolean inside) {
        last = current;
        current = ZoomingAuto;
        zoomingDirection = inside;
        zoomingAndMoving = false;
    }

    public void stopZooming() {
        last = current;
        if (!zoomingAndMoving)
            current = Waiting;
        else
            current = Moving;
        zoomingAndMoving = false;
    }

    public boolean isZooming() {
        return current == ZoomingAuto || zoomingAndMoving || current == ZoomingOnce;
    }

    public boolean getZoomingDirection() {
        if (!isZooming()) throw new IllegalStateException("cannot ask for zooming direction when not zooming");
        return zoomingDirection;
    }

    public void startMoving() {
        last = current;
        current = Moving;
    }

    public boolean isMoving() {
        return current == Moving || zoomingAndMoving;
    }

    public void stopMoving() {
        last = current;
        if (!zoomingAndMoving)
            current = Waiting;
        //else: current == Zooming{In,Out}, which is correct
        zoomingAndMoving = false;
    }

    public void startProgressiveRendering() {
        last = current;
        current = ProgressiveRendering;
        PRLvl = 0;
        zoomingAndMoving = false;
    }

    public int getProgressiveRenderingLevel() {
        if (!isProgressiveRendering())
            throw new IllegalStateException("cannot ask for Progressive rendering level when not Progressive rendering");
        return PRLvl;
    }

    public boolean isProgressiveRendering() {
        return current == ProgressiveRendering;
    }

    public boolean wasProgressiveRendering() {
        return last == ProgressiveRendering;
    }

    public boolean isWaiting() {
        return current == Waiting;
    }

    public RenderingMode getCurrent() {
        return current;
    }

    @Override
    public boolean isDifferentThanLast() {
        return current != last;
    }
}
