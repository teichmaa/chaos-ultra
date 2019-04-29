package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class OpenGLTexture {
    private OpenGLTextureHandle handle;
    private int target;

    public OpenGLTextureHandle getHandle() {
        return handle;
    }

    public int getTarget() {
        return target;
    }

    public OpenGLTexture(OpenGLTextureHandle handle, int target) {
        this.handle = handle;
        this.target = target;
    }

    public static OpenGLTexture of(OpenGLTextureHandle handle, int target){
        return new OpenGLTexture(handle, target);
    }
}
