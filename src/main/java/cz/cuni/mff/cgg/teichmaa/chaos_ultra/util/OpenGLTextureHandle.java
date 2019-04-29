package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class OpenGLTextureHandle {
    int value;

    public OpenGLTextureHandle(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static OpenGLTextureHandle of(int value) {
        return new OpenGLTextureHandle(value);
    }
}
