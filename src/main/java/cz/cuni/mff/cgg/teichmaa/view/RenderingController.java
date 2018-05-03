package cz.cuni.mff.cgg.teichmaa.view;

import com.jogamp.opengl.*;
import cz.cuni.mff.cgg.teichmaa.cuda.CudaLauncher;
import cz.cuni.mff.cgg.teichmaa.cuda.MandelbrotKernel;

import javax.swing.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;

import static com.jogamp.opengl.GL.*;
import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static com.jogamp.opengl.GL2ES3.GL_QUADS;
import static com.jogamp.opengl.fixedfunc.GLMatrixFunc.GL_MODELVIEW;

public class RenderingController extends MouseAdapter implements GLEventListener {
    //TODO bacha, tenhle listener běží v jiném vlákně než JavaFX - bacha na sdílené proměnné (a to samé tranzitivně pro CudaLauncher)

    private int outputTextureGLhandle;
    private CudaLauncher fractalRenderer;
    private JComponent owner;

    private int width;
    private int height;

    private float x = -0.5f;
    private float y = 0f;
    private float zoom = 2f;
    private int dwell = 2500;

    public RenderingController(int width, int height, JComponent owner) {
        this.width = width;
        this.height = height;
        this.owner = owner;
    }

    public void mouseWheelMoved(MouseWheelEvent e) {
        float coeff = 0.9f;
        if(e.getWheelRotation() > 0) coeff = 1.1f;
        zoom *= Math.pow(coeff, Math.abs(e.getPreciseWheelRotation()));
        recomputeKernelParams();
        owner.repaint();
    }

    private MouseEvent lastMousePosition;

    public void mouseDragged(MouseEvent e){
        if(lastMousePosition == null){
            return;
        }
        float dx = (lastMousePosition.getX() - e.getX()) * zoom / width ;
        float dy = (lastMousePosition.getY() - e.getY()) * zoom / height;
        x+=dx;
        y+=dy;
        recomputeKernelParams();
        lastMousePosition = e;
        owner.repaint();
    }
    public void mousePressed(MouseEvent e){
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
        fractalRenderer.getKernel().setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

    @Override
    public void init(GLAutoDrawable drawable) {
        final GL2 gl = drawable.getGL().getGL2();

        gl.glMatrixMode(GL_MODELVIEW);
        gl.glLoadIdentity();
        gl.glEnable(GL_TEXTURE_2D);

        int[] textureGLhandlePtr = new int[1];
        gl.glGenTextures(1, textureGLhandlePtr, 0);
        outputTextureGLhandle = textureGLhandlePtr[0];
        registerTexture(gl);

        fractalRenderer = new CudaLauncher(new MandelbrotKernel(0, width, height, 0, 0, 0, 0),
                outputTextureGLhandle);
        recomputeKernelParams();

    }

    private void registerTexture(GL gl) {
        //the following code is inspired by https://stackoverflow.com/questions/19244191/cuda-opengl-interop-draw-to-opengl-texture-with-cuda

        gl.glBindTexture(GL_TEXTURE_2D, outputTextureGLhandle);
        {
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null);
            //glTexImage2D params: target, level, internalFormat, width, height, border, format, type, data (may be null)
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

        fractalRenderer.launchKernel(false, true);

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
        width = i2;
        height = i3;

        final GL2 gl = drawable.getGL().getGL2();
        gl.glViewport(0, 0, width, height);

        fractalRenderer.getKernel().setWidth(width);
        fractalRenderer.getKernel().setHeight(height);

        fractalRenderer.unregisterOutputTexture();
        registerTexture(gl); //using the new dimensions
        fractalRenderer.registerOutputTexture(outputTextureGLhandle);

    }
}
