package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

/**
 * Wrapper class, containing an integer handle, representing a texture managed by OpenGL.
 */
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
