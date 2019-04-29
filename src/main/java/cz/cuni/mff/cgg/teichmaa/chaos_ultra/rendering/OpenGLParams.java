package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.OpenGLTexture;

public class OpenGLParams {
    private OpenGLTexture output;
    private OpenGLTexture palette;
    private int paletteLength;

    public OpenGLTexture getOutput() {
        return output;
    }

    public OpenGLTexture getPalette() {
        return palette;
    }

    public int getPaletteLength() {
        return paletteLength;
    }

    public OpenGLParams(OpenGLTexture output, OpenGLTexture palette, int paletteLength) {
        this.output = output;
        this.palette = palette;
        this.paletteLength = paletteLength;
    }

    public static OpenGLParams of(OpenGLTexture output, OpenGLTexture palette, int paletteLength) {
        return new OpenGLParams(output, palette, paletteLength);
    }
}
