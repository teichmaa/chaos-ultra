package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaFractalRenderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleMandelbrot;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;

import java.util.*;
import java.util.stream.Collectors;

public class FractalRendererProvider {

    private static final HashSet<Class<? extends FractalRenderingModule>> fractals = new HashSet<>();

    //register new fractals here:
    static {
        fractals.add(ModuleJulia.class);
        fractals.add(ModuleMandelbrot.class);
    }

    private FractalRenderingModule currentModule;
    private CudaFractalRenderer renderer;
    private final HashMap<String, FractalRenderingModule> activeModules = new HashMap<>();
    private final List<String> fractalsNames;

    public FractalRendererProvider(OpenGLParams glParams, ChaosUltraRenderingParams chaosParams) {
        this.glParams = glParams;
        this.chaosParams = chaosParams;

        fractalsNames = fractals.stream().map(Class::getSimpleName).collect(Collectors.toList());
    }

    private OpenGLParams glParams;
    private ChaosUltraRenderingParams chaosParams;


    public FractalRenderer getRenderer(String fractalName) {
        if (!fractalsNames.contains(fractalName)) {
            throw new IllegalArgumentException("Unknown fractal: " + fractalName);
        }
        if (!activeModules.containsKey(fractalName)) {
            createModule(fractalName);
        }

        currentModule = activeModules.get(fractalName);
        if(renderer!=null){
            renderer.close();
        }
        //TODO tohle je nejaky divny slozity, jak tu je vic rendereru a sdili spolu params. To je potreba jeste domyslet.
        renderer = new CudaFractalRenderer(currentModule, glParams,
                chaosParams);
        return renderer;
    }

    private void createModule(String fractalName) {
        if (!activeModules.containsKey(fractalName)) {
            FractalRenderingModule module = null;
            try {
                module = fractals.stream().filter(f -> f.getSimpleName().equals(fractalName)).findFirst().orElseThrow(IllegalArgumentException::new).newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                e.printStackTrace();
            }
            activeModules.put(fractalName, module);
        }
    }

    public List<String> getAvailableFractals() {
        return fractalsNames;
    }
}
