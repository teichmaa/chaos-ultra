package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class OpenGLTexture {
    private final OpenGLTextureHandle handle;
    private final int target;
    private final int width;
    private final int height;

    public OpenGLTextureHandle getHandle() {
        return handle;
    }

    public int getTarget() {
        return target;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public OpenGLTexture(OpenGLTextureHandle handle, int target, int width, int height) {
        this.handle = handle;
        this.target = target;
        this.width = width;
        this.height = height;
    }

    public static OpenGLTexture of(OpenGLTextureHandle handle, int target, int width, int height){
        return new OpenGLTexture(handle, target, width, height);
    }

    /**
     * Creates a new instance, with modified width and height. Original object stays unmodified.
     */
    public OpenGLTexture withNewSize(int width, int height){
        return new OpenGLTexture(this.getHandle(), this.getTarget(), width, height);
    }


}
