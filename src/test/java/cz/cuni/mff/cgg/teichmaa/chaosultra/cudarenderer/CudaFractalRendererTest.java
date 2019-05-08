package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer.modules.ModuleMandelbrot;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.GLParams;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

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

    }

    @Test
    public void cudaRendererInitializes(){
        FractalRenderingModule m = new ModuleMandelbrot();
        m.initialize();
        GLParams glParams = null; //todo
        CudaFractalRenderer r = new CudaFractalRenderer(m);
    }
}