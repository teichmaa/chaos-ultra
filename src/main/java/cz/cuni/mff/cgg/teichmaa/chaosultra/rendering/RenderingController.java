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

import static cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

public class RenderingController extends MouseAdapter {

    public static final double ZOOM_COEFF = 0.977f;

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
        repaintOnceLater.setRepeats(false);

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

    private Timer repaintOnceLater = new Timer(100, __ ->
            repaint());

    @Override
    public void mouseReleased(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();
        super.mouseReleased(e);

        animator.stop();
        if (SwingUtilities.isLeftMouseButton(e) && currentMode.isMoving()) {
            currentMode.stopMoving();
            repaintOnceLater.start();
        }
        if ((SwingUtilities.isRightMouseButton(e) || SwingUtilities.isMiddleMouseButton(e)) && currentMode.isZooming()) {
            currentMode.stopZooming();
            repaintOnceLater.start();
        }
    }

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

        model.setPlaneSegmentFromCenter(0, 0, 4);
        model.setMaxIterations(200);
        model.setSuperSamplingLevel(2);
        model.setUseAdaptiveSuperSampling(true);
        model.setUseFoveatedRendering(true);
        model.setUseSampleReuse(true);
        model.setAutomaticQuality(true);
        model.setVisualiseSampleCount(false);
        onModelUpdated();
        model.setFractalCustomParams("");

        repaint();
    }

    private void onModelUpdated() {
        guiPresenter.onModelUpdated(model.copy());
        {
            //special handling of the cuda initialization error that we want to show as a blocking popup
            Optional<String> cudaInitError = model.getNewlyLoggedErrors().stream().filter(e -> e.contains("CUDA installed?")).findFirst();
            cudaInitError.ifPresent(guiPresenter::showBlockingErrorAlertAsync);
        }
        model.getNewlyLoggedErrors().clear();
    }

    public void setPlaneSegmentRequested(double centerX, double centerY, double zoom) {
        assert SwingUtilities.isEventDispatchThread();
        //todo change automatic quality state?
        setPlaneSegment(centerX, centerY, zoom);
        onModelUpdated();
    }

    private void setPlaneSegment(double centerX, double centerY, double zoom) {
        model.setPlaneSegmentFromCenter(centerX, centerY, zoom);
    }

    public void setMaxIterationsRequested(int maxIterations) {
        assert SwingUtilities.isEventDispatchThread();
        //todo does this affect automatic quality?
        model.setMaxIterations(maxIterations);
        onModelUpdated();
    }

    /**
     * @param supSampLvl will be clamped to be >=1 and <= SUPER_SAMPLING_MAX_LEVEL
     */
    public void setSuperSamplingLevelRequested(int supSampLvl) {
        assert SwingUtilities.isEventDispatchThread();
        int newValue = Math.max(1, Math.min(supSampLvl, SUPER_SAMPLING_MAX_LEVEL));
        if(newValue != supSampLvl){
            System.out.println("Warning: super sampling level clamped to " + newValue + ", higher is not supported");
        }
        model.setSuperSamplingLevel(newValue);
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
        animator.stop();
        currentMode.resetState();
        showDefaultView();
        glView.onFractalChanged(fractalName);
        model.setSampleReuseCacheDirty(true);
        repaint();
    }

    public void debugRightBottomPixel() {
        glView.debugRightBottomPixel();
    }


    public void debugFractal() {
        glView.launchDebugKernel();
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
        startProgressiveRenderingAsync();
    }

    public void setUseSampleReuse(boolean value) {
        model.setUseSampleReuse(value);
    }

    public void startProgressiveRenderingAsync() {
        currentMode.startProgressiveRendering();
        repaint();
    }

    void onRenderingDone() {
        currentMode.step();
        if (currentMode.isProgressiveRendering())
            repaint();
        onModelUpdated();
    }

    public void saveImageRequested(String fileName, String format) {
        glView.saveImageAsync(fileName, format);
    }

    public GLEventListener getView() {
        return glView;
    }

}
