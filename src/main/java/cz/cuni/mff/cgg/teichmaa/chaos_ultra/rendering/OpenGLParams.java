package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;

public class OpenGLParams {
    private OpenGLTexture output;
    private OpenGLTexture palette;

    public OpenGLTexture getOutput() {
        return output;
    }

    public OpenGLTexture getPalette() {
        return palette;
    }

    /**
     * equal to getPalette.getWidth()
     */
    public int getPaletteLength() {
        return palette.getWidth();
    }

    public OpenGLParams(OpenGLTexture output, OpenGLTexture palette) {
        this.output = output;
        this.palette = palette;
    }

    public static OpenGLParams of(OpenGLTexture output, OpenGLTexture palette) {
        return new OpenGLParams(output, palette);
    }
}
