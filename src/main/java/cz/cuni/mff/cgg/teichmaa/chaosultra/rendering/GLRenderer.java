package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.awt.GLCanvas;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.CudaFractalRendererProvider;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.GLParams;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.GLTexture;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.GLTextureHandle;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.ImageHelpers;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.JavaHelpers;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.SimpleLogger;

import javax.swing.*;
import java.nio.Buffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static com.jogamp.opengl.GL.*;
import static cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.FractalRenderer.MAX_SUPER_SAMPLING;

class GLRenderer implements GLView {

    private static final String DEFAULT_COLOR_PALETTE_LOCATION = "palette.png";
    private static final String COLOR_PALETTE_PATH_PROPERTY_NAME = "colorPalette";
    public static final String RENDERING_LOGGING_PROPERTY_NAME = "renderingLogging";

    private GLTexture outputTexture;
    private GLTexture paletteTexture;
    private FractalRenderer fractalRenderer = new FractalRendererNullObjectVerbose(false);
    private FractalRendererProvider fractalRendererProvider;
    private final List<Consumer<GL2>> doBeforeDisplay = new ArrayList<>();
    private boolean doNotRenderCudaRequested = false;
    private final Model model;
    private final RenderingStateModel stateModel;
    private final RenderingController controller;
    private final GLCanvas target;
    private final SimpleLogger logger = new SimpleLogger();

    public GLRenderer(RenderingController controller, Model model, RenderingStateModel stateModel, GLCanvas target) {
        this.controller = controller;
        this.model = model;
        this.target = target;
        this.stateModel = stateModel;
        model.setCanvasWidth(target.getWidth());
        model.setCanvasHeight(target.getHeight());
        logger.setEnabled(
                Boolean.parseBoolean(System.getProperty(RENDERING_LOGGING_PROPERTY_NAME, "false"))
        );
    }

    @Override
    public void init(GLAutoDrawable drawable) {
        try {
            assert SwingUtilities.isEventDispatchThread();
            final GL2 gl = drawable.getGL().getGL2();

            //documentation for GL texture handling and lifecycle: https://www.khronos.org/opengl/wiki/Texture_Storage#Direct_creation
            int[] GLHandles = new int[2];
            gl.glGenTextures(GLHandles.length, GLHandles, 0);

            outputTexture = GLTexture.of(
                    GLTextureHandle.of(GLHandles[0]),
                    GL_TEXTURE_2D,
                    0,
                    0
            );

            String colorPalettePath = System.getProperty(COLOR_PALETTE_PATH_PROPERTY_NAME, DEFAULT_COLOR_PALETTE_LOCATION);
            Buffer colorPalette = IntBuffer.wrap(ImageHelpers.loadColorPaletteOrDefault(colorPalettePath));
            paletteTexture = GLTexture.of(
                    GLTextureHandle.of(GLHandles[1]),
                    GL_TEXTURE_2D,
                    colorPalette.limit(),
                    1
            );
            GLHelpers.specifyTextureSizeAndData(gl, paletteTexture, colorPalette);

            fractalRendererProvider = new CudaFractalRendererProvider();
            model.setAvailableFractals(fractalRendererProvider.getAvailableFractals());

            // This call can produce an exception.
            // However, fractalRenderer uses the null-object pattern, so even if not initialized properly to CudaFractalRenderer, we can still call its methods.
            fractalRenderer = fractalRendererProvider.getDefaultRenderer();
            model.setFractalName(fractalRenderer.getFractalName());

            controller.showDefaultView();
        } catch (Exception e) {
            model.logError(e.getMessage());
            if (JavaHelpers.isDebugMode()) throw e;
        }
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
        try {
            assert SwingUtilities.isEventDispatchThread();
            assert model.getCanvasWidth() == outputTexture.getWidth();
            assert model.getCanvasHeight() == outputTexture.getHeight();
            if (model.getCanvasWidth() == 0 || model.getCanvasHeight() == 0) {
                System.err.printf("Warning, RenderingController.display() called with width=%d, height=%d. Skipping the operation.\n", model.getCanvasWidth(), model.getCanvasHeight()); //todo make a logger for this
                return;
            }

            final GL2 gl = drawable.getGL().getGL2();
            try {
                doBeforeDisplay.forEach(c -> c.accept(gl));
            } finally {
                doBeforeDisplay.clear(); //do not repeat the functions if there has been an exception
            }
            long startTime = System.currentTimeMillis();

            if (stateModel.isZooming()) {
                controller.zoomAt(model.getLastMousePosition(), stateModel.getZoomingDirection());
            }

            if (doNotRenderCudaRequested) {
                doNotRenderCudaRequested = false; //reset the request state
                logger.debug("display() called with do not render requested");
            } else if (stateModel.isWaiting()) {
                /* nothing */
                logger.debug("display() called with stateModel.isWaiting(). Nothing done.");
            } else {
                boolean renderOK = cudaRender(drawable.getGL().getGL2());
                if (renderOK) {
                    long endTime = System.currentTimeMillis();
                    lastFrameRenderTime = (int) (endTime - startTime);
                    //lastFramesRenderTime.get(currentMode.getCurrent()).add((int) (endTime - startTime));
                    logger.logRenderingInfo("\t\t\t\tfinished in \t\t" + lastFrameRenderTime + " ms");
                    controller.onRenderingDone();
                } else {
                    logger.debug("rendering finished with not OK. (Probably end of progressive rendering)");
                }
            }
            GLHelpers.drawRectangle(gl, outputTexture);

        } catch (Exception e) {
            model.logError(e.getMessage());
            if (JavaHelpers.isDebugMode()) throw e;
        }
    }

    /**
     * @return true if OK, false if rendering should be canceled for some reason
     */
    private boolean cudaRender(final GL2 gl) {
        boolean qualityOK = determineRenderingModeQuality();
        if (!qualityOK)
            return false;

        model.setZooming(stateModel.isZooming());
        if (stateModel.isZooming())
            model.setZoomingIn(stateModel.getZoomingDirection());

        if (stateModel.isProgressiveRendering()) {
            fractalRenderer.renderQuality(model);
            logger.logRenderingInfo("\t\trender Quality, with super sampling set to \t\t\t" + model.getMaxSuperSampling());
            return true;
        } else {
            fractalRenderer.renderFast(model);
            logger.logRenderingInfo("\t\trender Fast, with SS \t\t\t\t" + model.getMaxSuperSampling());
            return true;
        }
    }

    /**
     * time in ms
     */
    private static final int shortestFrameRenderTime = 15;
    /**
     * time in ms
     */
    private static final int maxFrameRenderTime = 1000;
    private int lastFrameRenderTime = shortestFrameRenderTime;

    /**
     * @return true if OK, false if rendering should be canceled (quality to high)
     */
    private boolean determineRenderingModeQuality() {
        assert SwingUtilities.isEventDispatchThread();
        if (!model.isUseAutomaticQuality()) {
            return true;
        }

        logger.logRenderingInfo("currentMode: " + stateModel);

        if (stateModel.isDifferentThanLast()) {
            logger.logRenderingInfo("Automatic quality: RESET SS");
            model.setMaxSuperSampling(1);
            return true;
        }
        float prevSuperSampling = model.getMaxSuperSampling();

        if (stateModel.isZooming()) {
            setParamsToBeRenderedIn(shortestFrameRenderTime);
        } else if (stateModel.isMoving()) {
            setParamsToBeRenderedIn(shortestFrameRenderTime);
        } else if (stateModel.isProgressiveRendering()) {
            int desiredFrameRenderTime = shortestFrameRenderTime * 2 << stateModel.getProgressiveRenderingLevel(); //exponentially increasing the desired render time
            desiredFrameRenderTime = Math.max(lastFrameRenderTime * 2, desiredFrameRenderTime); //for cases when the last time, the actual frameRenderTime was much higher than desiredFrameRenderTime

            if (desiredFrameRenderTime > maxFrameRenderTime ||
                    model.getMaxSuperSampling() >= MAX_SUPER_SAMPLING
            ) {
                //if this is the maximal quality that we want to or can achieve, stop progressive rendering and do not render anymore
                if ((!stateModel.isProgressiveRendering() || stateModel.getProgressiveRenderingLevel() != 0)) { //level==0 happens upon parameter change -- don't stop in this case
                    stateModel.resetState();
                    model.setMaxSuperSampling(prevSuperSampling);
                    return false;
                }
            } else {
                setParamsToBeRenderedIn(desiredFrameRenderTime);
            }
        }
        return true;
    }

    private void setParamsToBeRenderedIn(int ms) {
        float newSS = model.getMaxSuperSampling() * ms / (float) lastFrameRenderTime;

        newSS = Math.min(newSS, MAX_SUPER_SAMPLING);
        model.setMaxSuperSampling(newSS);
        logger.logRenderingInfo("\t set params to be rendered in\t" + ms + " ms by setting maxSS to\t" + newSS + "");
    }

    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        try {
            final GL2 gl = drawable.getGL().getGL2();
            final int oldHeight = model.getCanvasHeight();
            model.setCanvasWidth(width);
            model.setCanvasHeight(height);
            model.setPlaneSegmentFromCenter(
                    model.getPlaneSegment().getCenterX(),
                    model.getPlaneSegment().getCenterY(),
                    model.getPlaneSegment().getZoom() * height / (double) oldHeight);
            if (oldHeight == 0) { //this happens during program initialization
                controller.showDefaultView(); //reinitialize current fractal
                fractalRenderer.supplyDefaultValues(model);
                fractalRenderer.setFractalCustomParams(model.getFractalCustomParams());
            }

            if (fractalRenderer.getState() == FractalRendererState.readyToRender)
                fractalRenderer.freeRenderingResources();
            outputTexture = outputTexture.withNewSize(width, height);
            GLHelpers.specifyTextureSize(gl, outputTexture);
            fractalRenderer.initializeRendering(GLParams.of(outputTexture, paletteTexture));

            controller.startProgressiveRenderingAsync();
        } catch (Exception e) {
            model.logError(e.getMessage());
            if (JavaHelpers.isDebugMode()) throw e;
        }
    }

    @Override
    public void saveImageAsync(String fileName, String format) {
        assert SwingUtilities.isEventDispatchThread();
        doBeforeDisplay.add(gl -> {
            int width_t = model.getCanvasWidth();
            int height_t = model.getCanvasHeight();
            int[] data = new int[width_t * height_t];
            Buffer b = IntBuffer.wrap(data);
            gl.glBindTexture(GL_TEXTURE_2D, outputTexture.getHandle().getValue());
            {
                //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTexImage.xhtml
                gl.glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, b);
            }
            gl.glBindTexture(GL_TEXTURE_2D, 0);
            ImageHelpers.saveImageToFile(data, width_t, height_t, fileName, format);
            logger.logRenderingInfo("Image saved to " + fileName);
            doNotRenderCudaRequested = true;
        });
        repaint();
    }

    @Override
    public void onFractalCustomParamsUpdated() {
        try {
            fractalRenderer.setFractalCustomParams(model.getFractalCustomParams());
            model.setSampleReuseCacheDirty(true);
        } catch (Exception e) {
            model.logError(e.getMessage());
            if (JavaHelpers.isDebugMode()) throw e;
        }
    }

    @Override
    public void repaint() {
        target.repaint();
    }

    @Override
    public void onFractalChanged(String fractalName, boolean forceReload) {
        doBeforeDisplay.add(gl -> {
            if (fractalRenderer.getState() == FractalRendererState.readyToRender)
                fractalRenderer.freeRenderingResources();
            fractalRenderer = fractalRendererProvider.getRenderer(fractalName, forceReload);
            fractalRenderer.supplyDefaultValues(model);
            fractalRenderer.setFractalCustomParams(model.getFractalCustomParams());
            fractalRenderer.initializeRendering(GLParams.of(outputTexture, paletteTexture));
            controller.showDefaultView();
        });
        repaint();
    }

    @Override
    public void launchDebugKernel() {
        fractalRenderer.launchDebugKernel();
    }

    @Override
    public void showDefaultView() {
        fractalRenderer.supplyDefaultValues(model);
        fractalRenderer.setFractalCustomParams(model.getFractalCustomParams());
        repaint();
    }
}
