package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;
import cz.cuni.mff.cgg.teichmaa.chaosultra.gui.GUIPresenter;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointInt;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.util.Optional;

import static cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRenderer.MAX_SUPER_SAMPLING;

public class RenderingController extends MouseAdapter {

    public static double ZOOM_COEFF = 0.9995f;
    public static double DEEPSETVALUE = 0.9995f;

    private static RenderingController singleton = null;

    private final Animator animator;

    private final GUIPresenter guiPresenter;
    private final GLView glView;

    private final Model model = new Model();
    private final RenderingModeFSM currentMode = new RenderingModeFSM();

    public RenderingController(GLCanvas target, GUIPresenter guiPresenter) {
        if (singleton == null) {
            singleton = this;
        } else throw new IllegalStateException("Cannot instantiate more than one RenderingController");

        this.guiPresenter = guiPresenter;
        glView = new GLRenderer(this, model, currentMode, target);
        animator = new Animator(target);
        animator.setRunAsFastAsPossible(true);
        animator.stop();
        startProgressiveRenderingLater.setRepeats(false);

//        for(RenderingModeFSM.RenderingMode mode : RenderingModeFSM.RenderingMode.values()){
//            lastFramesRenderTime.put(mode, new CyclicBuffer(lastFramesRenderTimeBufferLength, shortestFrameRenderTime));
//        }

        model.setErrorLoggedCallback(() -> guiPresenter.onModelUpdated(model.copy()));
    }

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        assert SwingUtilities.isEventDispatchThread();
        super.mouseWheelMoved(e);

        model.getLastMousePosition().setXYFrom(e);
        model.getMouseFocus().setXYFrom(e);
        currentMode.doZoomingManualOnce(e.getWheelRotation() < 0);
        repaint();
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        super.mouseMoved(e);
        model.getMouseFocus().setXYFrom(e);
        model.getLastMousePosition().setXYFrom(e);
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();
        super.mouseDragged(e);


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
        super.mousePressed(e);

        model.getLastMousePosition().setXYFrom(e);
        model.getMouseFocus().setXYFrom(e);
        if(animator.isAnimating()){
            animator.stop();
            if (currentMode.isMoving()) {
                currentMode.stopMoving();
            }
            if (currentMode.isZooming()) {
                currentMode.stopZooming();
            }
        }else {
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
    }

    private Timer startProgressiveRenderingLater = new Timer(100, __ -> {
        currentMode.startProgressiveRendering();
        repaint();
    });

//    @Override
//    public void mouseReleased(MouseEvent e) {
//        assert SwingUtilities.isEventDispatchThread();
//        super.mouseReleased(e);
//
//        animator.stop();
//        if (SwingUtilities.isLeftMouseButton(e) && currentMode.isMoving()) {
//            currentMode.stopMoving();
//            startProgressiveRenderingLater.start();
//        }
//        if ((SwingUtilities.isRightMouseButton(e) || SwingUtilities.isMiddleMouseButton(e)) && currentMode.isZooming()) {
//            currentMode.stopZooming();
//            startProgressiveRenderingLater.start();
//        }
//    }

    /**
     * Apply zooming to model values, especially the plane segment
     *
     * @param where texture coordinates of the zooming center
     * @param into  whether to zoom in or out
     */
    public void zoomAt(PointInt where, boolean into) {
        double segment_width = model.getPlaneSegment().getSegmentWidth();
        double segment_height = model.getPlaneSegment().getSegmentHeight();

        double relTop = where.getY() / (double) model.getCanvasHeight(); //relative distance from zoomingCenter to border, \in (0,1)
        double relBtm = 1 - relTop;
        double relLeft = where.getX() / (double) model.getCanvasWidth();
        double relRght = 1 - relLeft;

        double center_x = model.getPlaneSegment().getLeftBottom().getX() + segment_width * relLeft;
        double center_y = model.getPlaneSegment().getLeftBottom().getY() + segment_height * relBtm;

        double zoom_coeff = into ? RenderingController.ZOOM_COEFF : 2f - RenderingController.ZOOM_COEFF;

        double l_b_new_x = center_x - segment_width * relLeft * zoom_coeff;
        double l_b_new_y = center_y - segment_height * relBtm * zoom_coeff;
        double r_t_new_x = center_x + segment_width * relRght * zoom_coeff;
        double r_t_new_y = center_y + segment_height * relTop * zoom_coeff;

        model.getPlaneSegment().setAll(l_b_new_x, l_b_new_y, r_t_new_x, r_t_new_y);
    }

    public void showDefaultView() {
        animator.stop();
        currentMode.resetState();
        currentMode.startProgressiveRendering();

        model.resetRenderingValuesToDefault();
        glView.showDefaultView();

        onModelUpdated();
    }

    private void onModelUpdated() {
        guiPresenter.onModelUpdated(model.copy());
        {
            //special handling of the cuda initialization error that we want to show as a blocking popup
            Optional<String> cudaInitError = model.getNewlyLoggedErrors().stream()
                    .filter(e -> e.contains("CUDA installed?") || e.contains("CUDA_ERROR_NO_DEVICE"))
                    .findFirst();
            cudaInitError.ifPresent(guiPresenter::showBlockingErrorAlertAsync);
        }
        model.getNewlyLoggedErrors().clear();
        if (currentMode.isWaiting())
            currentMode.startProgressiveRendering();

    }

    public void setPlaneSegmentRequested(double centerX, double centerY, double zoom) {
        assert SwingUtilities.isEventDispatchThread();
        setPlaneSegment(centerX, centerY, zoom);
        onModelUpdated();
    }

    private void setPlaneSegment(double centerX, double centerY, double zoom) {
        model.setPlaneSegmentFromCenter(centerX, centerY, zoom);
    }

    public void setMaxIterationsRequested(int maxIterations) {
        assert SwingUtilities.isEventDispatchThread();
        model.setMaxIterations(maxIterations);
        onModelUpdated();
    }

    /**
     * @param supSampLvl will be clamped to be >=1 and <= SUPER_SAMPLING_MAX_LEVEL
     */
    public void setMaxSuperSamplingRequested(float supSampLvl) {
        assert SwingUtilities.isEventDispatchThread();
        float newValue = Math.min(supSampLvl, MAX_SUPER_SAMPLING);
        if(newValue != supSampLvl){
            System.out.println("Warning: super sampling level clamped to " + newValue + ", higher is not supported");
        }
        model.setMaxSuperSampling(newValue);
        onModelUpdated();
    }

    public void repaint() {
        glView.repaint();
    }

    public void setFractalCustomParams(String text) {
        assert SwingUtilities.isEventDispatchThread();
        model.setFractalCustomParams(text);
        glView.onFractalCustomParamsUpdated();
        repaint();
    }

    public void onFractalChanged(String fractalName) {
        assert SwingUtilities.isEventDispatchThread();
        if(fractalName == null || fractalName.isEmpty())
            return;
        if(fractalName.equals(model.getFractalName()))
            return;
        model.setFractalName(fractalName);
        showDefaultView();
        glView.onFractalChanged(fractalName, false);
        model.setSampleReuseCacheDirty(true);
        showDefaultView();
        repaint();
    }

    public void debugFractal() {
        glView.launchDebugKernel();
    }

    public void setVisualiseSampleCount(boolean value) {
        model.setVisualiseSampleCount(value);
        startProgressiveRenderingAsync();
        repaint();
    }

    public void setUseAdaptiveSuperSampling(boolean value) {
        model.setUseAdaptiveSuperSampling(value);
    }

    public void setAutomaticQuality(boolean value) {
        model.setAutomaticQuality(value);
    }

    public void setUseFoveatedRendering(boolean value) {
        model.setUseFoveatedRendering(value);
    }

    public void setUseSampleReuse(boolean value) {
        model.setUseSampleReuse(value);
    }

    public void startProgressiveRenderingAsync() {
        currentMode.resetState();
        currentMode.startProgressiveRendering();
        repaint();
    }

    void onRenderingDone() {
        currentMode.step();
        if (currentMode.isProgressiveRendering() && model.isUseAutomaticQuality())
            repaint();
        onModelUpdated();
    }

    public void saveImageRequested(String fileName, String format) {
        glView.saveImageAsync(fileName, format);
    }

    public GLEventListener getView() {
        return glView;
    }

    public void reloadFractal() {
        glView.onFractalChanged(model.getFractalName(), true);
        model.setSampleReuseCacheDirty(true);
        repaint();
    }
}
