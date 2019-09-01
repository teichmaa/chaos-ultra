package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

import java.util.Set;

/**
 * The provider should catalog available fractals and be able to list them.
 * <br />
 * The provider is responsible for creating appropriate instances of FractalRenderer
 * and for disposing the old ones.
 */
public interface FractalRendererProvider {

    /**
     * @return Returns an implementation of some fractal renderer. Usually, this is a renderer of the Mandelbrot set.
     */
    FractalRenderer getDefaultRenderer();

    /**
     * Creates or re-uses a fractal renderer and disposes the current one, if a new has been created.
     * @param fractalName name of the fractal, choosen from the set returned by {@code getAvailableFractals}
     * @param forceReload force renderer reload even if asking a renderer that has already been loaded
     * @return a renderer implementation
     */
    FractalRenderer getRenderer(String fractalName, boolean forceReload);

    Set<String> getAvailableFractals();
}
