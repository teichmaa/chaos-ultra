package cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui;

import com.jogamp.opengl.*;

import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.*;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params.RenderingModel;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.GLHelpers;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTextureHandle;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.nio.Buffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static com.jogamp.opengl.GL.*;
import static cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

public class RenderingController extends MouseAdapter implements GLEventListener {

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

    private OpenGLTexture outputTexture;
    private OpenGLTexture paletteTexture;
    private FractalRenderer fractalRenderer = new FractalRendererNullObjectVerbose();
    private FractalRendererProvider fractalRendererProvider;
    private GLCanvas target;
    private ControllerFX controllerFX;
    private Animator animator;
    private RenderingModeFSM currentMode = new RenderingModeFSM();

    private int width_t; //todo toto je mozna obsolete, protoze ty hodnoty jsou v modelu
    private int height_t;

    private RenderingModel model = new RenderingModel();
    private List<Consumer<GL2>> doBeforeDisplay = new ArrayList<>();
    private boolean doNotRenderRequested = false;

    public RenderingController(GLCanvas target, ControllerFX controllerFX) {
        this.width_t = target.getWidth();
        this.height_t = target.getHeight();
        this.target = target;
        this.controllerFX = controllerFX;
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

    private MouseEvent lastMousePosition;
    private PointInt focus = new PointInt();

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        lastMousePosition = e;
        focus.setXYFrom(e);
        currentMode.doZoomingManualOnce(e.getWheelRotation() < 0);
        target.repaint();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        if (SwingUtilities.isLeftMouseButton(e)) {
            if (lastMousePosition == null) {
                return;
            }
            double textureToZoomCoeff = model.getSegment().getSegmentHeight() / height_t;
            double dx = (lastMousePosition.getX() - e.getX()) * textureToZoomCoeff;
            double dy = -(lastMousePosition.getY() - e.getY()) * textureToZoomCoeff;
            model.getSegment().increaseXsBy(dx);
            model.getSegment().increaseYsBy(dy);

            currentMode.startMoving();
            target.repaint();
        }
        if (!currentMode.isMoving())
            focus.setXYFrom(e);
        lastMousePosition = e;
    }

    @Override
    public void mousePressed(MouseEvent e) {
        assert SwingUtilities.isEventDispatchThread();

        lastMousePosition = e;
        focus.setXYFrom(e);
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
            target.repaint());

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

    private void zoomAt(MouseEvent e, boolean into) {
        zoomAt(e.getX(), e.getY(), into);
    }

    /**
     * @param texture_x zooming center, texture x-coordinate
     * @param texture_y zooming center, texture y-coordinate
     * @param into      whether to zoom in or out
     */
    private void zoomAt(int texture_x, int texture_y, boolean into) {
        double segment_width = model.getSegment().getSegmentWidth();
        double segment_height = model.getSegment().getSegmentHeight();

        double relTop = texture_y / (double) height_t; //relative distance from zoomingCenter to border, \in (0,1)
        double relBtm = 1 - relTop;
        double relLeft = texture_x / (double) width_t;
        double relRght = 1 - relLeft;

        double center_x = model.getSegment().getLeftBottom().getX() + segment_width * relLeft;
        double center_y = model.getSegment().getLeftBottom().getY() + segment_height * relBtm;

        double zoom_coeff = this.ZOOM_COEFF;
        if (!into) zoom_coeff = 2f - this.ZOOM_COEFF; //todo refactor

        double l_b_new_x = center_x - segment_width * relLeft * zoom_coeff;
        double l_b_new_y = center_y - segment_height * relBtm * zoom_coeff;
        double r_t_new_x = center_x + segment_width * relRght * zoom_coeff;
        double r_t_new_y = center_y + segment_height * relTop * zoom_coeff;

        model.getSegment().setAll(l_b_new_x, l_b_new_y, r_t_new_x, r_t_new_y);
    }

    @Override
    public void init(GLAutoDrawable drawable) {
        assert SwingUtilities.isEventDispatchThread();
        final GL2 gl = drawable.getGL().getGL2();

        //documentation for GL texture handling and lifecycle: https://www.khronos.org/opengl/wiki/Texture_Storage#Direct_creation
        int[] GLHandles = new int[2];
        gl.glGenTextures(GLHandles.length, GLHandles, 0);

        outputTexture = OpenGLTexture.of(
                OpenGLTextureHandle.of(GLHandles[0]),
                GL_TEXTURE_2D,
                0,
                0
        );

        Buffer colorPalette = IntBuffer.wrap(ImageHelpers.createColorPalette());
        paletteTexture = OpenGLTexture.of(
                OpenGLTextureHandle.of(GLHandles[1]),
                GL_TEXTURE_2D,
                colorPalette.limit(),
                1
        );
        GLHelpers.specifyTextureSizeAndData(gl, paletteTexture, colorPalette);

        fractalRendererProvider = new FractalRendererProvider(model);

//        fractalRenderer = fractalRendererProvider.getRenderer("mandelbrot");
        fractalRenderer = fractalRendererProvider.getRenderer("julia");
        // fractalRenderer uses the null-object pattern, so even if not initialized properly to CudaFractalRenderer, we can still call its methods
        //todo domyslet jak je to s tim null objectem a kdo vlastne vyhazuje jakou vyjimku kdy (modul vs renderer) a jak ji zobrazovat a kdo je zopovedny za ten null object a tyhle shity

        showDefaultView();
    }

    public void showDefaultView() {
        animator.stop();
        currentMode.reset();

        updateSegment(-0.5, 0, 2);
        model.setMaxIterations(800);
        model.setSuperSamplingLevel(5);
        model.setUseAdaptiveSuperSampling(true);
        model.setUseFoveatedRendering(true);
        model.setUseSampleReuse(true);
        model.setAutomaticQuality(true);
        model.setVisualiseSampleCount(false);
        controllerFX.onModelUpdated(model);

        repaint();
    }

    @Override
    public void dispose(GLAutoDrawable drawable) {
        assert SwingUtilities.isEventDispatchThread();
        if (fractalRenderer == null) return;
        fractalRenderer.close();

        final GL2 gl = drawable.getGL().getGL2();
        int[] textures = new int[2];
        textures[0] = outputTexture.getHandle().getValue();
        textures[1] = paletteTexture.getHandle().getValue();
        gl.glDeleteTextures(textures.length, textures, 0);
    }

    @Override
    public void display(GLAutoDrawable drawable) {
        assert SwingUtilities.isEventDispatchThread();
        assert width_t == outputTexture.getWidth();
        assert height_t == outputTexture.getHeight();
        if (width_t == 0 || height_t == 0) {
            System.err.printf("Warning, RenderingController.display() called with width=%d, height=%d. Skipping the operation.\n", width_t, height_t); //todo make a logger for this
            return;
        }

        final GL2 gl = drawable.getGL().getGL2();
        doBeforeDisplay.forEach(c -> c.accept(gl));
        doBeforeDisplay.clear();
        if (doNotRenderRequested) {
            doNotRenderRequested = false;
            return;
        }

        long startTime = System.currentTimeMillis();

        //todo tohle cele by asi ocenilo lepsi navrh. Inspiruj se u her a u Update smycky atd
        //  ve finale by kernel launch asi mohl nebyt blocking
        //      trivialni napad: double buffering, GL vzdy vykresli tu druhou, nez cuda zrovna pocita

        if (currentMode.isZooming()) {
            zoomAt(lastMousePosition, currentMode.getZoomingDirection());
        }

        updateQuality();
        render(drawable.getGL().getGL2());

        currentMode.step();
        if (currentMode.isProgressiveRendering())
            this.repaint();

        controllerFX.onModelUpdated(model.copy());

        long endTime = System.currentTimeMillis();
        lastFrameRenderTime = (int) (endTime - startTime);
        //lastFramesRenderTime.get(currentMode.getCurrent()).add((int) (endTime - startTime));
//        System.out.println("" + lastFrameRenderTime + " ms (frame total render time)");
    }

    private static final int shortestFrameRenderTime = 15;
    private static final int maxFrameRenderTime = 1000;
    private static final int lastFramesRenderTimeBufferLength = 2;
    //private Map<RenderingModeFSM.RenderingMode, CyclicBuffer> lastFramesRenderTime = new HashMap<>();
    private int lastFrameRenderTime = shortestFrameRenderTime;

    private void updateQuality() {
        assert SwingUtilities.isEventDispatchThread();
        if (!model.isAutomaticQuality()) return;
        if (currentMode.isWaiting() && currentMode.wasProgressiveRendering()) {
            model.setSuperSamplingLevel(10); //todo lol proc zrovna deset, kde se to vzalo?
            return;
        }

        //System.out.println("currentMode = " + currentMode);
        if (currentMode.isZooming()) {
            setParamsToBeRenderedIn(shortestFrameRenderTime);
        } else if (currentMode.isMoving()) {
            setParamsToBeRenderedIn(shortestFrameRenderTime);
        } else if (currentMode.isWaiting()) {
            setParamsToBeRenderedIn(shortestFrameRenderTime * 2);
        } else if (currentMode.isProgressiveRendering()) {
            int desiredFrameRenderTime = shortestFrameRenderTime * 2 << currentMode.getProgressiveRenderingLevel();
            if (desiredFrameRenderTime > maxFrameRenderTime)
                currentMode.reset();
            else
                setParamsToBeRenderedIn(desiredFrameRenderTime);
            //pridat sem currentMode.getHighQualityIteration()
            //   a do RenderingMode::step dat highQIteration++
        }
        if (model.getSuperSamplingLevel() == SUPER_SAMPLING_MAX_LEVEL)
            currentMode.reset();
    }

    private void setParamsToBeRenderedIn(int ms) {
        //debug:
//        System.out.print(currentMode + ": ");
//        CyclicBuffer b = lastFramesRenderTime.get(currentMode.getCurrent());
//        for (int i = 0; i < lastFramesRenderTimeBufferLength; i++) {
//            System.out.print(b.get(i) + "\t");
//        }
//        System.out.println();

        //int mean = Math.round(lastFramesRenderTime.get(currentMode.getCurrent()).getMeanValue());
        int newSS = Math.round(model.getSuperSamplingLevel() * ms / (float) Math.max(1, lastFrameRenderTime));
        newSS = Math.max(1, Math.min(newSS, SUPER_SAMPLING_MAX_LEVEL));
        model.setSuperSamplingLevel(newSS);
    }

    private void render(final GL2 gl) {
        if (currentMode.wasProgressiveRendering()) {
            fractalRenderer.renderQuality(model);
        } else {
            model.setFocus(focus);
            model.setZooming(currentMode.isZooming());
            fractalRenderer.renderFast(model);
        }

        GLHelpers.drawRectangle(gl, outputTexture);
    }

    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        final GL2 gl = drawable.getGL().getGL2();
        final int oldHeight = height_t;
        this.width_t = width;
        this.height_t = height;
        model.setCanvasWidth(width);
        model.setCanvasHeight(height);

        updateSegment(
                model.getSegment().getCenterX(),
                model.getSegment().getCenterY(),
                model.getSegment().getZoom() * height / (double) oldHeight);

        if (fractalRenderer.getState() == FractalRendererState.readyToRender)
            fractalRenderer.freeRenderingResources();
        outputTexture = outputTexture.withNewSize(width, height);
        GLHelpers.specifyTextureSize(gl, outputTexture);
        fractalRenderer.initializeRendering(OpenGLParams.of(outputTexture, paletteTexture));

        currentMode.startProgressiveRendering();
    }

    public void setBoundsRequested(double center_x, double center_y, double zoom) {
        assert SwingUtilities.isEventDispatchThread();
        //todo change automatic quality state?
        updateSegment(center_x, center_y, zoom);
        controllerFX.onModelUpdated(model.copy());
    }

    private void updateSegment(double center_x, double center_y, double zoom) {
        double windowHeight = 1;
        double windowWidth = windowHeight / (double) height_t * width_t;
        double segment_left_bottom_x = center_x - windowWidth * zoom / 2;
        double segment_left_bottom_y = center_y - windowHeight * zoom / 2;
        double segment_right_top_x = center_x + windowWidth * zoom / 2;
        double segment_right_top_y = center_y + windowHeight * zoom / 2;
        model.getSegment().setAll(segment_left_bottom_x, segment_left_bottom_y, segment_right_top_x, segment_right_top_y);
    }

    void setMaxIterationsRequested(int maxIterations) {
        assert SwingUtilities.isEventDispatchThread();
        //todo does this affect automatic quality?
        model.setMaxIterations(maxIterations);
        controllerFX.onModelUpdated(model.copy());
    }

    void setSuperSamplingLevelRequested(int supSampLvl) {
        assert SwingUtilities.isEventDispatchThread();
        //supSampLvl will be clamped to be >=1 and <= SUPER_SAMPLING_MAX_LEVEL
        int newValue = Math.max(1, Math.min(supSampLvl, SUPER_SAMPLING_MAX_LEVEL));
        if(newValue != supSampLvl){
            System.out.println("Warning: super sampling level clamped to " + newValue + ", higher is not supported");
        }
        model.setSuperSamplingLevel(newValue);
        controllerFX.onModelUpdated(model.copy());
    }

    void repaint() {
        target.repaint();
    }

    public void saveImage(String fileName, String format) {
        assert SwingUtilities.isEventDispatchThread();
        doBeforeDisplay.add(gl -> {
            int[] data = new int[width_t * height_t];
            Buffer b = IntBuffer.wrap(data);
            gl.glBindTexture(GL_TEXTURE_2D, outputTexture.getHandle().getValue());
            {
                //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTexImage.xhtml
                gl.glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, b);
            }
            gl.glBindTexture(GL_TEXTURE_2D, 0);
            ImageHelpers.createImage(data, width_t, height_t, fileName, format);
            System.out.println("Image saved to " + fileName);
            doNotRenderRequested = true;
        });
        repaint();
    }

    public void setFractalSpecificParams(String text) {
        assert SwingUtilities.isEventDispatchThread();
        fractalRenderer.setFractalSpecificParams(text);
        repaint();
    }

    public void onFractalChanged(String fractalName) {
        assert SwingUtilities.isEventDispatchThread();
        doBeforeDisplay.add(gl -> {
            animator.stop();
            currentMode.reset();

            fractalRenderer.freeRenderingResources();
            fractalRenderer = fractalRendererProvider.getRenderer(fractalName);
            fractalRenderer.initializeRendering(OpenGLParams.of(outputTexture, paletteTexture));
        });
        showDefaultView();
        repaint();
    }

    public void debugRightBottomPixel() {
        fractalRenderer.debugRightBottomPixel();
    }


    public void debugFractal() {
        fractalRenderer.launchDebugKernel();
    }
}
