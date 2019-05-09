package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

/**
 * Data class representing a texture managed by OpenGL.
 */
public class GLTexture {
    private final GLTextureHandle handle;
    private final int target;
    private final int width;
    private final int height;

    public GLTextureHandle getHandle() {
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

    public GLTexture(GLTextureHandle handle, int target, int width, int height) {
        this.handle = handle;
        this.target = target;
        this.width = width;
        this.height = height;
    }

    public static GLTexture of(GLTextureHandle handle, int target, int width, int height){
        return new GLTexture(handle, target, width, height);
    }

    /**
     * Creates a new instance, with modified width and height. Original object stays unmodified.
     */
    public GLTexture withNewSize(int width, int height){
        return new GLTexture(this.getHandle(), this.getTarget(), width, height);
    }


}
