package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

public class GLParams {
    private GLTexture output;
    private GLTexture palette;

    public GLTexture getOutput() {
        return output;
    }

    public GLTexture getPalette() {
        return palette;
    }

    /**
     * equal to getPalette.getWidth()
     */
    public int getPaletteLength() {
        return palette.getWidth();
    }

    public GLParams(GLTexture output, GLTexture palette) {
        this.output = output;
        this.palette = palette;
    }

    public static GLParams of(GLTexture output, GLTexture palette) {
        return new GLParams(output, palette);
    }
}
