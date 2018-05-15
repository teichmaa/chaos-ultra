package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;

import com.jogamp.opengl.*;

import com.jogamp.opengl.awt.GLCanvas;
import com.jogamp.opengl.util.Animator;
import cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer.FractalRenderer;
import cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer.MandelbrotKernel;

import javafx.application.Platform;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.nio.Buffer;
import java.nio.IntBuffer;

import static com.jogamp.opengl.GL.*;
import static com.jogamp.opengl.GL2.GL_QUADS;
import static com.jogamp.opengl.fixedfunc.GLMatrixFunc.GL_MODELVIEW;
import static cz.cuni.mff.cgg.teichmaa.mandelzoomer.view.RenderingMode.*;

public class RenderingController extends MouseAdapter implements GLEventListener {

    private static RenderingController singleton = null;

    static RenderingController getSingleton() {
        return singleton;
    }
    public static final int SUPER_SAMPLING_MAX_LEVEL = FractalRenderer.SUPER_SAMPLING_MAX_LEVEL;

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

    private int outputTextureGLhandle;
    private int paletteTextureGLhandle;
    private FractalRenderer fractalRenderer;
    private GLCanvas target;
    private ControllerFX controllerFX;
    private Animator animator;
    private RenderingMode currentMode = Nothing;

    private int width_t;
    private int height_t;

    private float plane_left_bottom_x;
    private float plane_left_bottom_y;
    private float plane_right_top_x;
    private float plane_right_top_y;

    public RenderingController(GLCanvas target, ControllerFX controllerFX) {
        this.width_t = target.getWidth();
        this.height_t = target.getHeight();
        this.target = target;
        this.controllerFX = controllerFX;
        animator = new Animator(target);
        animator.setRunAsFastAsPossible(true);
        animator.stop();
        currentMode = Nothing;
        singleton = this;
    }

    private MouseEvent lastMousePosition;

    @Override
    public void mouseWheelMoved(MouseWheelEvent e) {
        nothingHasHappened = false;
        zoomAt(e, e.getWheelRotation() < 0);
        target.repaint();
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        nothingHasHappened = false;
        if (SwingUtilities.isLeftMouseButton(e)) {
            if (lastMousePosition == null) {
                return;
            }
            float textureToZoomCoeff = getPlaneHeight() / height_t;
            float dx = (lastMousePosition.getX() - e.getX()) * textureToZoomCoeff;
            float dy = -(lastMousePosition.getY() - e.getY()) * textureToZoomCoeff;
            plane_right_top_x += dx;
            plane_right_top_y += dy;
            plane_left_bottom_x += dx;
            plane_left_bottom_y += dy;

            target.repaint();
        }
        lastMousePosition = e;
    }

    @Override
    public void mousePressed(MouseEvent e) {
        lastMousePosition = e;
        nothingHasHappened = false;
        if (SwingUtilities.isRightMouseButton(e)) {
            currentMode = ZoomingIn;
            animator.start();
        }
    }

    private boolean nothingHasHappened = false;
    private Timer renderHQAfterWaiting = new Timer(100, __ -> {
        if(nothingHasHappened){
            currentMode = HighQuality;
            nothingHasHappened = false;
            target.repaint();
        }
    });

    @Override
    public void mouseReleased(MouseEvent e) {
        if (SwingUtilities.isRightMouseButton(e) && (currentMode== ZoomingIn || currentMode == ZoomingOut)) {
            animator.stop();
            currentMode = Balanced;
            nothingHasHappened = true;
            renderHQAfterWaiting.setRepeats(false);
            renderHQAfterWaiting.start();
        }
    }

    private static final float ZOOM_COEFF = 0.977f;

    private void zoomAt(MouseEvent e, boolean into) {
        zoomAt(e.getX(), e.getY(), into);
    }

    /**
     * @param texture_x zooming center, texture x-coordinate
     * @param texture_y zooming center, texture y-coordinate
     * @param into      whether to zoom in or out
     */
    private void zoomAt(int texture_x, int texture_y, boolean into) {
        float plane_width = getPlaneWidth();
        float plane_height = getPlaneHeight();

        float relTop = texture_y / (float) height_t; //relative distance from zoomingCenter to border, \in (0,1)
        float relBtm = 1 - relTop;
        float relLeft = texture_x / (float) width_t;
        float relRght = 1 - relLeft;

        float plane_x = plane_left_bottom_x + plane_width * relLeft;
        float plane_y = plane_left_bottom_y + plane_height * relBtm;

        float zoom_coeff = this.ZOOM_COEFF;
        if (!into) zoom_coeff = 2f - this.ZOOM_COEFF;

        float l_b_new_x = plane_x - plane_width * relLeft * zoom_coeff;
        float l_b_new_y = plane_y - plane_height * relBtm * zoom_coeff;
        float r_t_new_x = plane_x + plane_width * relRght * zoom_coeff;
        float r_t_new_y = plane_y + plane_height * relTop * zoom_coeff;

        this.plane_left_bottom_x = l_b_new_x;
        this.plane_left_bottom_y = l_b_new_y;
        this.plane_right_top_x = r_t_new_x;
        this.plane_right_top_y = r_t_new_y;
    }

    private void updateKernelParams() {
        fractalRenderer.setBounds(plane_left_bottom_x, plane_left_bottom_y, plane_right_top_x, plane_right_top_y);
    }

    private void updateFXUI() {
        Platform.runLater(() -> {
            controllerFX.setZoom(getZoom());
            controllerFX.setX(getCenterX());
            controllerFX.setY(getCenterY());
            controllerFX.setDwell(fractalRenderer.getDwell());
            controllerFX.setSuperSamplingLevel(fractalRenderer.getSuperSamplingLevel());
        });
    }

    @Override
    public void init(GLAutoDrawable drawable) {
        final GL2 gl = drawable.getGL().getGL2();

        gl.glMatrixMode(GL_MODELVIEW);
        gl.glLoadIdentity();
        gl.glEnable(GL_TEXTURE_2D);

        int[] GLhandles = new int[2];
        gl.glGenTextures(GLhandles.length, GLhandles, 0);
        outputTextureGLhandle = GLhandles[0];
        paletteTextureGLhandle = GLhandles[1];
        registerOutputTexture(gl);
        Buffer colorPalette = IntBuffer.wrap(ImageHelpers.createColorPalette());
        gl.glBindTexture(GL_TEXTURE_2D, paletteTextureGLhandle);
        {
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, colorPalette.limit(), 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, colorPalette);
        }
        gl.glBindTexture(GL_TEXTURE_2D, 0);

        fractalRenderer = new FractalRenderer(new MandelbrotKernel(),
                outputTextureGLhandle, GL_TEXTURE_2D, paletteTextureGLhandle, GL_TEXTURE_2D, colorPalette.limit());

        Platform.runLater(controllerFX::showDefaultView);
        currentMode = Balanced;
    }

    private void registerOutputTexture(GL gl) {
        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        {
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width_t, height_t, 0, GL_RGBA, GL_UNSIGNED_BYTE, null);
            //glTexImage2D params: target, level, internalFormat, width, height, border (must 0), format, type, data (may be null)
            //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        }
        gl.glBindTexture(GL_TEXTURE_2D, 0);
    }

    @Override
    public void dispose(GLAutoDrawable drawable) {
        fractalRenderer.close();
    }

    @Override
    public void display(GLAutoDrawable drawable) {

        long startTime = System.currentTimeMillis();
        if (currentMode == Nothing)
            return;

        //todo tohle cele by asi ocenilo lepsi navrh. Inspiruj se u her a u Update smycky atd
        //  ve finale by kernel launch asi mohl nebyt blocking
        //      trivialni napad: double buffering, GL vzdy vykresli tu druhou, nez cuda zrovna pocita

        if (saveImageRequested) {
            saveImageInternal(drawable.getGL().getGL2());
            saveImageRequested = false;
            return;
        }

        if (currentMode == ZoomingIn)
            zoomAt(lastMousePosition, true);
        else if (currentMode == ZoomingOut)
            zoomAt(lastMousePosition, false);

        if(currentMode != Manual)
            updateQuality();
        updateKernelParams();
        updateFXUI();
        render(drawable.getGL().getGL2());

        if(currentMode == HighQuality) //HQ is one-time use only
            currentMode = Balanced;

        long endTime = System.currentTimeMillis();
        lastFrameRenderTime = (int) (endTime - startTime);
        //System.out.println("" + lastFrameRenderTime + " ms (frame total render time)");
    }

    private final int idealFrameRenderTime = 15;
    private int lastFrameRenderTime = idealFrameRenderTime;

    private void updateQuality() {
        if (currentMode == ZoomingIn || currentMode == ZoomingOut) {
            setParamsToBeRenderedIn(idealFrameRenderTime);
        }
        if(currentMode == Balanced){
            setParamsToBeRenderedIn(idealFrameRenderTime * 4);
        }
        if(currentMode == HighQuality){
            setParamsToBeRenderedIn(idealFrameRenderTime * 10);
        }
    }

    private void setParamsToBeRenderedIn(int ms){
        //todo tohle musi byt prumer poslednich nekolika, jinak to bude blikat
        int newSS = fractalRenderer.getSuperSamplingLevel() * ms / Math.max(1, lastFrameRenderTime);
        System.out.println("newSS = " + newSS);
        setSuperSamplingLevel(newSS);
    }

    private void render(final GL2 gl) {
        fractalRenderer.launchKernel(false, false);

        gl.glMatrixMode(GL_MODELVIEW);
        //gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        gl.glBegin(GL_QUADS);
        {
            //map screen quad to texture quad and make it render
            gl.glTexCoord2f(0f, 1f);
            gl.glVertex2f(-1f, -1f);
            gl.glTexCoord2f(1f, 1f);
            gl.glVertex2f(+1f, -1f);
            gl.glTexCoord2f(1f, 0f);
            gl.glVertex2f(+1f, +1f);
            gl.glTexCoord2f(0f, 0f);
            gl.glVertex2f(-1f, +1f);
        }
        gl.glEnd();
        //gl.glBindTexture(GL_TEXTURE_2D, 0);
        //gl.glPopMatrix();
        //gl.glFinish();
    }

    @Override
    public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
        final GL2 gl = drawable.getGL().getGL2();
        int oldHeight = height_t;
        this.width_t = width;
        this.height_t = height;
        setBounds(getCenterX(), getCenterY(), getZoom() * height / (float) oldHeight);

        fractalRenderer.unregisterOutputTexture();
        registerOutputTexture(gl); //already using the new dimensions
        fractalRenderer.resize(width, height, outputTextureGLhandle, GL_TEXTURE_2D);
        //repaint();
    }

    private float getPlaneWidth() {
        return plane_right_top_x - plane_left_bottom_x;
    }

    private float getPlaneHeight() {
        return plane_right_top_y - plane_left_bottom_y;
    }

    float getCenterX() {
        return plane_left_bottom_x + getPlaneWidth() / 2;
    }

    float getCenterY() {
        return plane_left_bottom_y + getPlaneHeight() / 2;
    }

    float getZoom() {
        return plane_right_top_y - plane_left_bottom_y;
    }

    void setBounds(float center_x, float center_y, float zoom) {
        float windowHeight = 1;
        float windowWidth = windowHeight / (float) height_t * width_t;
        plane_left_bottom_x = center_x - windowWidth * zoom / 2;
        plane_left_bottom_y = center_y - windowHeight * zoom / 2;
        plane_right_top_x = center_x + windowWidth * zoom / 2;
        plane_right_top_y = center_y + windowHeight * zoom / 2;
    }

    void setDwell(int dwell) {
        fractalRenderer.setDwell(dwell);
    }

    void setSuperSamplingLevel(int supSampLvl) {
        //supSampLvl will be clamped to be >=1 and <= SUPER_SAMPLING_MAX_LEVEL
        fractalRenderer.setSuperSamplingLevel(Math.max(1,Math.min(supSampLvl, SUPER_SAMPLING_MAX_LEVEL)));
    }

    void repaint() {
        currentMode = Manual;
        target.repaint();
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        fractalRenderer.setAdaptiveSS(adaptiveSS);
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        fractalRenderer.setVisualiseAdaptiveSS(visualiseAdaptiveSS);
    }

    public void saveImage(String fileName, String format) {
        this.saveImageFileName = fileName;
        this.saveImageFormat = format;
        saveImageRequested = true;
        repaint();
    }

    private String saveImageFileName;
    private String saveImageFormat;
    private boolean saveImageRequested = false;

    private void saveImageInternal(GL2 gl) {
        int[] data = new int[width_t * height_t];
        Buffer b = IntBuffer.wrap(data);
        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        {
            //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetTexImage.xhtml
            gl.glGetTexImage(GL_TEXTURE_2D, 0, GL_BGRA, GL_UNSIGNED_BYTE, b);
        }
        gl.glBindTexture(GL_TEXTURE_2D, 0);
        ImageHelpers.createImage(data, width_t, height_t, saveImageFileName, saveImageFormat);
        System.out.println("Image saved to " + saveImageFileName);
    }
}
