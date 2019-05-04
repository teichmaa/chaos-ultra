package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui.ControllerFX;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui.GUIController;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.util.Collection;
import java.util.Set;

import static cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

public class RenderingController extends MouseAdapter {

    private static RenderingController singleton = null;

    // terminology of private fields:
    //   texture: {int x int} discrete set, representing the surface that we draw on
    //   plane (aka complex plane): {real x real} continuous set, representing the part of the complex plane that we render
    //   name_x: x-coordinate of a point in texture or Real coordinate of a point in complex plane
    //   name_y: y-coordinate of a point in texture or Imag coordinate of a point in complex plane
    //   attributeName_t: texture attribute
    //   attributeName_p: plane attribute

    //        Texture (int x int)        Complex plane (real x real)
    //      (0,0)    __ __
    //        | x > |__|__|             .
    //        |y |__|__|__|             .
    //        |v |__|__|__|    <==>     .
    //        |__|__|__|__|             ^
    //        |__|__|__|__|             y
    //        |__|__|__|__|           (0,0) x > .......


    private final Animator animator;

    private final GUIController guiController;
    private final GLRenderer glRenderer;

    private final Model model = new Model();
    private final RenderingModeFSM currentMode = new RenderingModeFSM();

    public RenderingController(GLCanvas target, GUIController guiController) {
        this.guiController = guiController;
        glRenderer = new GLRenderer(this, model,currentMode,target);
        animator = new Animator(target);
        animator.setRunAsFastAsPossible(true);
        animator.stop();
        renderInFuture.setRepeats(false);

//        for(RenderingModeFSM.RenderingMode mode : RenderingModeFSM.RenderingMode.values()){
//            lastFramesRenderTime.put(mode, new CyclicBuffer(lastFramesRenderTimeBufferLength, shortestFrameRenderTime));
//        }

        if (singleton == null)
            singleton = this;
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        model.getLastMousePosition().setXYFrom(e);
        model.getMouseFocus().setXYFrom(e);
        currentMode.doZoomingManualOnce(e.getWheelRotation() < 0);
        repaint();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        if (SwingUtilities.isLeftMouseButton(e)) {
            double canvasToZoomCoeff = model.getPlaneSegment().getSegmentHeight() / model.getCanvasHeight();
            double dx = (model.getLastMousePosition().getX() - e.getX()) * canvasToZoomCoeff;
            double dy = -(model.getLastMousePosition().getY() - e.getY()) * canvasToZoomCoeff;
            model.getPlaneSegment().increaseXsBy(dx);
            model.getPlaneSegment().increaseYsBy(dy);

            currentMode.startMoving();
            repaint();
        }
        if (!currentMode.isMoving())
            model.getMouseFocus().setXYFrom(e);
        model.getLastMousePosition().setXYFrom(e);
    }

    @Override
    public void mousePressed(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        model.getLastMousePosition().setXYFrom(e);
        model.getMouseFocus().setXYFrom(e);
        if (SwingUtilities.isRightMouseButton(e) && SwingUtilities.isLeftMouseButton(e)) {
            currentMode.startZoomingAndMoving(true);
            animator.start();
        } else if (SwingUtilities.isRightMouseButton(e)) {
            currentMode.startZooming(true);
            animator.start();
        } else if (SwingUtilities.isLeftMouseButton(e)) {
            //currentMode.startMoving();
            //animator.start();
        } else if (SwingUtilities.isMiddleMouseButton(e)) {
            currentMode.startZooming(false);
            animator.start();
        }
    }

    private Timer renderInFuture = new Timer(100, __ ->
            repaint());

    @Override
    public void mouseReleased(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        animator.stop();
        if (SwingUtilities.isLeftMouseButton(e) && currentMode.isMoving()) {
            currentMode.stopMoving();
            renderInFuture.start();
        }
        if ((SwingUtilities.isRightMouseButton(e) || SwingUtilities.isMiddleMouseButton(e)) && currentMode.isZooming()) {
            currentMode.stopZooming();
            renderInFuture.start();
        }
    }

    private static final double ZOOM_COEFF = 0.977f;

    public void zoomAt(PointInt where, boolean into) {
        zoomAt(where.getX(), where.getY(), into);
    }

    /**
     * @param texture_x zooming center, texture x-coordinate
     * @param texture_y zooming center, texture y-coordinate
     * @param into      whether to zoom in or out
     */
    private void zoomAt(int texture_x, int texture_y, boolean into) {
        double segment_width = model.getPlaneSegment().getSegmentWidth();
        double segment_height = model.getPlaneSegment().getSegmentHeight();

        double relTop = texture_y / (double) model.getCanvasHeight(); //relative distance from zoomingCenter to border, \in (0,1)
        double relBtm = 1 - relTop;
        double relLeft = texture_x / (double) model.getCanvasWidth();
        double relRght = 1 - relLeft;

        double center_x = model.getPlaneSegment().getLeftBottom().getX() + segment_width * relLeft;
        double center_y = model.getPlaneSegment().getLeftBottom().getY() + segment_height * relBtm;

        double zoom_coeff = this.ZOOM_COEFF;
        if (!into) zoom_coeff = 2f - this.ZOOM_COEFF; //todo refactor

        double l_b_new_x = center_x - segment_width * relLeft * zoom_coeff;
        double l_b_new_y = center_y - segment_height * relBtm * zoom_coeff;
        double r_t_new_x = center_x + segment_width * relRght * zoom_coeff;
        double r_t_new_y = center_y + segment_height * relTop * zoom_coeff;

        model.getPlaneSegment().setAll(l_b_new_x, l_b_new_y, r_t_new_x, r_t_new_y);
    }

    public void showDefaultView() {
        animator.stop();
        currentMode.resetState();

        model.setPlaneSegmentFromCenter(-0.5, 0, 2);
        model.setMaxIterations(800);
        model.setSuperSamplingLevel(5);
        model.setUseAdaptiveSuperSampling(true);
        model.setUseFoveatedRendering(true);
        model.setUseSampleReuse(true);
        model.setAutomaticQuality(true);
        model.setVisualiseSampleCount(false);
        guiController.onModelUpdated(model.copy());

        repaint();
    }

    public void setPlaneSegmentRequested(double centerX, double centerY, double zoom) {
        assert SwingUtilities.isEventDispatchThread();
        //todo change automatic quality state?
        setPlaneSegment(centerX, centerY, zoom);
        guiController.onModelUpdated(model.copy());
    }

    private void setPlaneSegment(double centerX, double centerY, double zoom) {
        model.setPlaneSegmentFromCenter(centerX, centerY, zoom);
    }

    public void setMaxIterationsRequested(int maxIterations) {
        assert SwingUtilities.isEventDispatchThread();
        //todo does this affect automatic quality?
        model.setMaxIterations(maxIterations);
        guiController.onModelUpdated(model.copy());
    }

    public void setSuperSamplingLevelRequested(int supSampLvl) {
        assert SwingUtilities.isEventDispatchThread();
        //supSampLvl will be clamped to be >=1 and <= SUPER_SAMPLING_MAX_LEVEL
        int newValue = Math.max(1, Math.min(supSampLvl, SUPER_SAMPLING_MAX_LEVEL));
        if(newValue != supSampLvl){
            System.out.println("Warning: super sampling level clamped to " + newValue + ", higher is not supported");
        }
        model.setSuperSamplingLevel(newValue);
        guiController.onModelUpdated(model.copy());
    }

    public void repaint() {
        glRenderer.repaint();
    }

    public void setFractalSpecificParams(String text) {
        assert SwingUtilities.isEventDispatchThread();
        glRenderer.setFractalSpecificParams(text);
        repaint();
    }

    public void onFractalChanged(String fractalName) {
        assert SwingUtilities.isEventDispatchThread();
        if(fractalName == null || fractalName.isEmpty())
            return;
        if(fractalName.equals(model.getFractalName()))
            return;
        model.setFractalName(fractalName);
        animator.stop();
        currentMode.resetState();
        glRenderer.onFractalChanged(fractalName);
        showDefaultView();
        model.setSampleReuseCacheDirty(true);
        repaint();
    }

    public void debugRightBottomPixel() {
        glRenderer.debugRightBottomPixel();
    }


    public void debugFractal() {
        glRenderer.launchDebugKernel();
    }

    public void setVisualiseSampleCount(boolean value) {
        model.setVisualiseSampleCount(value);
        repaint();
        currentMode.startProgressiveRendering();
    }

    public void setUseAdaptiveSuperSampling(boolean value) {
        model.setUseAdaptiveSuperSampling(value);
    }

    public void setAutomaticQuality(boolean value) {
        model.setAutomaticQuality(value);
    }

    public void setUseFoveatedRendering(boolean value) {
        model.setUseFoveatedRendering(value);
        startProgressiveRendering();
    }

    public void setUseSampleReuse(boolean value) {
        model.setUseSampleReuse(value);
    }

    public void saveImage(String fileName, String format) {
        glRenderer.saveImage(fileName, format);
    }

    public GLEventListener getView() {
        return glRenderer;
    }

    void onRenderingDone(){
        guiController.onModelUpdated(model.copy());
    }


    public void startProgressiveRendering() {
        currentMode.startProgressiveRendering();
        repaint();
    }

}
