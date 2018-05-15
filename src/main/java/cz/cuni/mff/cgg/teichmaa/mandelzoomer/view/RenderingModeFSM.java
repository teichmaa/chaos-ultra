package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;


import com.jogamp.graph.curve.opengl.RenderState;

import java.awt.event.MouseEvent;

import static cz.cuni.mff.cgg.teichmaa.mandelzoomer.view.RenderingModeFSM.RenderingMode.*;

/***
 * Finite State Machine for Rendering Mode transitions
 */
class RenderingModeFSM {

    enum RenderingMode {
        Balanced,
        ZoomingIn,
        ZoomingOut,
        Moving,
        HighQuality;
    }

    @Override
    public String toString() {
        String state = !zoomingAndMoving ? current.toString() : "ZoomingAndMoving";
        return "FSM in state " + state;
    }

    private RenderingMode current = Balanced;
    private RenderingMode last;

    private boolean zoomingAndMoving = false;

    void step() {
        RenderingMode newValue = current;
        if (current == Balanced && (last == ZoomingIn || last == ZoomingOut || last == Moving))
            newValue = HighQuality;
        else if (current == HighQuality)
            newValue = Balanced;
        //default: do nothing
        last = current;
        current = newValue;
    }

    void startZoomingAndMoving(boolean inside) {
        startZooming(inside);
        zoomingAndMoving = true;

    }

    void startZooming(boolean inside) {
        last = current;
        if (inside)
            current = ZoomingIn;
        else
            current = ZoomingOut;
        zoomingAndMoving = false;
    }

    void stopZooming() {
        last = current;
        if(!zoomingAndMoving)
            current = Balanced;
        else
            current = Moving;
        zoomingAndMoving = false;
    }

    boolean isZooming(){
        return current == ZoomingOut || current == ZoomingIn || zoomingAndMoving;
    }

    boolean getZoomingDirection(){
        if(!isZooming()) throw new IllegalStateException("cannot ask for zooming direction when not zooming");
        return current == ZoomingIn;
    }

    void startMoving(){
        last = current;
        current = Moving;
        zoomingAndMoving = false;
    }

    boolean isMoving(){
        return current == Moving || zoomingAndMoving;
    }

    void stopMoving(){
        last = current;
        if(!zoomingAndMoving)
            current = Balanced; //zooming will be kept set
        zoomingAndMoving = false;
    }

    boolean isHighQuality(){
        return current == HighQuality;
    }

    boolean isBalanced(){
        return current == Balanced;
    }

    RenderingMode getCurrent(){
        return current;
    }

}
