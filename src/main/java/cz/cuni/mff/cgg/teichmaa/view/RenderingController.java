package cz.cuni.mff.cgg.teichmaa.view;

import com.jogamp.opengl.*;
import cz.cuni.mff.cgg.teichmaa.cuda.AbstractFractalRenderKernel;
import cz.cuni.mff.cgg.teichmaa.cuda.CudaLauncher;
import cz.cuni.mff.cgg.teichmaa.cuda.MandelbrotKernel;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.nio.Buffer;
import java.nio.IntBuffer;

import static com.jogamp.opengl.GL.*;
import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static com.jogamp.opengl.GL2.GL_QUADS;
import static com.jogamp.opengl.fixedfunc.GLMatrixFunc.GL_MODELVIEW;

public class RenderingController extends MouseAdapter implements GLEventListener {
    //TODO bacha, tenhle listener běží v jiném vlákně než JavaFX - bacha na sdílené proměnné (a to samé tranzitivně pro CudaLauncher)

    private int outputTextureGLhandle;
    private int paletteTextureGLhandle;
    private CudaLauncher fractalRenderer;
    private JComponent owner;

    private int width;
    private int height;

    private float x = -0.5f;
    private float y = 0f;
    private float zoom = 2f;
    private int dwell = 1200;
    private int superSamplingLevel = 32;

    public RenderingController(int width, int height, JComponent owner) {
        this.width = width;
        this.height = height;
        this.owner = owner;
    }

    public void mouseWheelMoved(MouseWheelEvent e) {
        float coeff = 0.95f;
        if (e.getWheelRotation() > 0) coeff = 2f - coeff;
        zoom *= Math.pow(coeff, Math.abs(e.getPreciseWheelRotation()));
        recomputeKernelParams();
        owner.repaint();
    }

    private MouseEvent lastMousePosition;

    public void mouseDragged(MouseEvent e) {
        if (lastMousePosition == null) {
            return;
        }
        float dx = (lastMousePosition.getX() - e.getX()) * zoom / width;
        float dy = (lastMousePosition.getY() - e.getY()) * zoom / height;
        x += dx;
        y += dy;
        recomputeKernelParams();
        lastMousePosition = e;
        owner.repaint();
    }

    public void mousePressed(MouseEvent e) {
        lastMousePosition = e;
    }

    private void recomputeKernelParams() {

        float windowHeight = 1;
        float windowWidth = windowHeight / (float) height * width;
        float left_bottom_x = x - windowWidth * zoom / 2;
        float left_bottom_y = y - windowHeight * zoom / 2;
        float right_top_x = x + windowWidth * zoom / 2;
        float right_top_y = y + windowHeight * zoom / 2;

        fractalRenderer.getKernel().setDwell(dwell);
        fractalRenderer.getKernel().setSuperSamplingLevel(superSamplingLevel);
        fractalRenderer.getKernel().setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

    public AbstractFractalRenderKernel getKernel_unsafe(){
        return this.fractalRenderer.getKernel();
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

        fractalRenderer = new CudaLauncher(new MandelbrotKernel(0, width, height, 0, 0, 0, 0),
                outputTextureGLhandle, paletteTextureGLhandle, colorPalette.limit());
        recomputeKernelParams();

    }

    private void registerOutputTexture(GL gl) {
        //the following code is inspired by https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda

        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        {
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null);
            //glTexImage2D params: target, level, internalFormat, width, height, border (must 0), format, type, data (may be null)
            //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        }
        gl.glBindTexture(GL_TEXTURE_2D, 0);
    }

    @Override
    public void dispose(GLAutoDrawable drawable) {

    }

    @Override
    public void display(GLAutoDrawable drawable) {

        long startTime = System.currentTimeMillis();

        fractalRenderer.launchKernel(false, false);

        final GL2 gl = drawable.getGL().getGL2();
        gl.glMatrixMode(GL_MODELVIEW);
        gl.glPushMatrix();
        gl.glLoadIdentity();
        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        gl.glBegin(GL_QUADS);
        {
            gl.glTexCoord2f(0f, 0f);
            gl.glVertex2f(-1.0f, -1.0f);
            gl.glTexCoord2f(1f, 0f);
            gl.glVertex2f(1.0f, -1.0f);
            gl.glTexCoord2f(1f, 1f);
            gl.glVertex2f(1.0f, 1.0f);
            gl.glTexCoord2f(0f, 1f);
            gl.glVertex2f(-1.0f, 1.0f);
        }
        gl.glEnd();
        gl.glPopMatrix();
        gl.glFinish();

        long endTime = System.currentTimeMillis();
        System.out.println("display() function finished in " + (endTime - startTime) + " ms");

    }

    @Override
    public void reshape(GLAutoDrawable drawable, int i, int i1, int i2, int i3) {
        /*
        //TODO je potreba reshapovat i supersamplovaci random vzorek
        width = i2;
        height = i3;

        final GL2 gl = drawable.getGL().getGL2();
        gl.glViewport(0, 0, width, height);

        fractalRenderer.getKernel().setWidth(width);
        fractalRenderer.getKernel().setHeight(height);

        fractalRenderer.unregisterOutputTexture();
        registerOutputTexture(gl); //using the new dimensions
        fractalRenderer.registerOutputTexture(outputTextureGLhandle);
*/
    }
}
