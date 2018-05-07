package cz.cuni.mff.cgg.teichmaa.view;


import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;

public class ConcurrentParamHolder {

    private final List lockDims = new ArrayList();
    private final List lockXY = new ArrayList();

    public ConcurrentParamHolder() {
    }

    private int width;
    private int height;
    private float x = -0.5f;
    private float y = 0f;
    private float zoom = 2f;
    private int dwell = 1200;
    private int superSamplingLevel = 32;
    private boolean requestingRender = false;
    private boolean adaptiveSS = true;
    private boolean visualiseAdaptiveSS = true;

    public boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    public void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
    }

    public boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    public void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
    }

    public boolean isRequestingRender() {
        return requestingRender;
    }

    public void setRequestingRender(boolean requestingRender) {
        this.requestingRender = requestingRender;
    }

    public synchronized int getWidth() {
        synchronized (lockDims) {
            return width;
        }
    }

    public synchronized int getHeight() {
        synchronized (lockDims) {
            return height;
        } //todo porad spatne, w a h muzou byt nekonzistentni :(
    }

    public void setDimensions(int width, int height) {
        if (isRequestingRender()) return;
        synchronized (lockDims) {
            this.width = width;
            this.height = height;
        }
    }

    public float getX() {
        synchronized (lockXY) {
            return x;
        }
    }

    public float getY() {
        synchronized (lockXY) {
            return y;
        }
    }

    public void setXY(float x, float y) {
        if (isRequestingRender()) return;
        synchronized (lockXY) {
            this.x = x;
            this.y = y;
        }
    }

    public synchronized float getZoom() {
        return zoom;
    }

    public synchronized void setZoom(float zoom) {
        if(isRequestingRender()) return;
        this.zoom = zoom;
    }

    public synchronized int getDwell() {
        return dwell;
    }

    public synchronized void setDwell(int dwell) {
        if(isRequestingRender()) return;
        this.dwell = dwell;
    }

    public synchronized int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

    public synchronized void setSuperSamplingLevel(int superSamplingLevel) {
        if(isRequestingRender()) return;
        this.superSamplingLevel = superSamplingLevel;
    }
}
