package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.GLParams;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.RenderingModel;

import java.io.Closeable;

/**
 * Represents a class that is able, using some underlying technology, to sample a fractal and render it on an OpenGL texture.
 * <br />
 * Each instance may represent a different fractal.
 */
public interface FractalRenderer extends Closeable {
    int MAX_SUPER_SAMPLING = 64;

    /**
     * Initializes rendering and set the renderer to readyToRender state.
     * The implementation is allowed (and expected) to map the textures to its internal resources, potentially locking the textures for modification and for exclusive writing.
     * <br />
     * The resources have to be freed by calling {@code freeRenderingResources} or {@code close}.
     * <br />
     * May be called only when a corresponding OpenGL context is active (e.g. during GLEventListener events).
     * <br />
     * Using this stateful approach rather stateless (registering the resource every time a frame is rendered) is for performance reasons.
     *
     * @param glParams glParams containing the output texture to render on and the color palette
     */
    void initializeRendering(GLParams glParams);

    /**
     * Frees the textures previously supplied to {@code initializeRendering}, allowing for texture modification, and sets the renderer to notInitialized state. The renderer may be later initialized again.
     */
    void freeRenderingResources();

    FractalRendererState getState();

    int getWidth();

    int getHeight();

    void launchDebugKernel();

    /**
     * Should use any available means to reduce the computational complexity of fractal sampling.
     * <br />
     * The implementation is expected be scalable; in the sense that the lower the values of {@code maxIterations} and {@code maxSuperSampling}, the faster computation time.
     * @throws FractalRendererException upon rendering error
     * @param model model with data to render
     */
    void renderFast(RenderingModel model);

    /**
     * Should provide high quality images with little artifacts, at the cost of taking longer time to compute.
     * <br />
     * The implementation is expected be scalable; in the sense that the higher the values of {@code maxIterations} and {@code maxSuperSampling}, the higher visual quality.
     * @throws FractalRendererException upon rendering error
     * @param model model with data to render
     */
    void renderQuality(RenderingModel model);

    /**
     * Closes the object, calling {@code freeRenderingResources} beside others.
     */
    @Override
    void close();

    void setFractalCustomParams(String text);

    String getFractalName();

    /**
     * Asks the underlying implementation to provide default values to be set before starting to render the fractal. This is an optional operation and may be left empty.
     * @param model model to supply the default values to
     */
    void supplyDefaultValues(DefaultFractalModel model);
}
