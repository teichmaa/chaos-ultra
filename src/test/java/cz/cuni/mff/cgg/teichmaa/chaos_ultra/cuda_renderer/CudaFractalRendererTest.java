package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.GLEventListener;
import com.jogamp.opengl.awt.GLCanvas;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleMandelbrot;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.OpenGLParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTextureHandle;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import javax.swing.*;

import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static org.junit.jupiter.api.Assertions.*;

class CudaFractalRendererTest {

    @BeforeAll
    public static void testInit(){
        System.setProperty("cudaKernelsDir","src\\main\\cuda");
    }

    @Test
    public void cudaInitializes(){
        FractalRenderingModule m = new ModuleMandelbrot();
    }

    @Test
    public void moduleInitializes(){
        FractalRenderingModule m = new ModuleMandelbrot();
        m.initialize();
    }

    @Test
    public void twoModulesInitialize(){
        FractalRenderingModule m = new ModuleMandelbrot();
        m.initialize();
        FractalRenderingModule j = new ModuleJulia();
        j.initialize();
    }

    @Test
    public void openGLInitializes(){
        GLCanvas canvas = new GLCanvas();
        OpenGLTexture outputTexture;
        GLEventListener listener = new GLEventListener() {
            @Override
            public void init(GLAutoDrawable glAutoDrawable) {
//                GLContext ctx = GLContext.getCurrent();
//                assertNotNull(ctx);
//
//                final GL2 gl = glAutoDrawable.getGL().getGL2();
//                int[] GLHandles = new int[2];
//                gl.glGenTextures(GLHandles.length, GLHandles, 0);
//                outputTexture = OpenGLTexture.of(OpenGLTextureHandle.of(GLHandles[0]), GL_TEXTURE_2D);
//                registerOutputTexture(gl);
//                paletteTexture = OpenGLTexture.of(OpenGLTextureHandle.of(GLHandles[1]), GL_TEXTURE_2D);

            }

            @Override
            public void dispose(GLAutoDrawable glAutoDrawable) {

            }

            @Override
            public void display(GLAutoDrawable glAutoDrawable) {

            }

            @Override
            public void reshape(GLAutoDrawable glAutoDrawable, int i, int i1, int i2, int i3) {

            }
        };
        canvas.addGLEventListener(listener);

    }

    @Test
    public void cudaRendererInitializes(){
        FractalRenderingModule m = new ModuleMandelbrot();
        m.initialize();
        OpenGLParams glParams = null; //todo
        ChaosUltraRenderingParams chaosParams = null; //todo
        CudaFractalRenderer r = new CudaFractalRenderer(m, glParams, chaosParams);
    }
}