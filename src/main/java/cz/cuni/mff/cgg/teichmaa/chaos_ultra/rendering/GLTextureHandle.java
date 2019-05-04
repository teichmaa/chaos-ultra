package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

public class GLTextureHandle {
    int value;

    public GLTextureHandle(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static GLTextureHandle of(int value) {
        return new GLTextureHandle(value);
    }
}
