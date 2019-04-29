package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaFractalRenderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.Point2DDoubleImmutable;

import static com.jogamp.opengl.GL.GL_TEXTURE_2D;

public class FractalRendererProvider {

    public FractalRendererProvider(int outputTextureGLHandle, int outputTextureGLTarget, int paletteTextureGLHandle, int paletteTextureGLTarget, int paletteLength, ChaosUltraRenderingParams params) {
        this.outputTextureGLHandle = outputTextureGLHandle;
        this.outputTextureGLTarget = outputTextureGLTarget;
        this.paletteTextureGLHandle = paletteTextureGLHandle;
        this.paletteTextureGLTarget = paletteTextureGLTarget;
        this.paletteLength = paletteLength;
        this.params = params;
    }

    private int outputTextureGLHandle;
    private int outputTextureGLTarget;
    private int paletteTextureGLHandle;
    private int paletteTextureGLTarget;
    private int paletteLength;
    private ChaosUltraRenderingParams params;

    private ModuleJulia julia;
    private FractalRenderer renderer;

    public FractalRenderer getRenderer() {
        julia = new ModuleJulia(Point2DDoubleImmutable.of(0, 0.5));
        renderer = new CudaFractalRenderer(julia, outputTextureGLHandle, outputTextureGLTarget, paletteTextureGLHandle, paletteTextureGLTarget, paletteLength,
                params);
        return renderer;
    }

    public void setFractalSpecificParams(String text) {
        String[] tokens = text.split(",");
        double x = Double.parseDouble(tokens[0].trim());
        double y = Double.parseDouble(tokens[1].trim());
        julia.setC(Point2DDoubleImmutable.of(x, y));
    }
}
