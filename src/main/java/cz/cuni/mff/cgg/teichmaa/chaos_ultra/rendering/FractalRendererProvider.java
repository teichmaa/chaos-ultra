package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaFractalRenderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaHelpers;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaInitializationException;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleMandelbrot;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.Newton;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class FractalRendererProvider {

    private static final HashSet<Class<? extends FractalRenderingModule>> modules = new HashSet<>();
    private final Map<String, FractalRenderingModule> moduleInstances;

    static {
        //register new fractals here:
        modules.add(ModuleJulia.class);
        modules.add(ModuleMandelbrot.class);
        modules.add(Newton.class);

        //end register section

    }

    public FractalRendererProvider() {
        moduleInstances = modules.stream().map(FractalRendererProvider::createInstance).collect(Collectors.toMap(FractalRenderingModule::getFractalName, Function.identity()));
    }

    public Set<String> getAvailableFractals() {
        return moduleInstances.keySet();
    }

    private CudaFractalRenderer activeRenderer;

    public FractalRenderer getDefaultRenderer(){
        return getRenderer("mandelbrot");
    }

    public FractalRenderer getRenderer(String fractalName) {
        if(activeRenderer !=null){
            if(activeRenderer.getFractalName().equals(fractalName))
                return activeRenderer; //returning the current active renderer
            else{
                activeRenderer.close(); //closing the previous renderer
            }
        }

        if (!moduleInstances.containsKey(fractalName)) {
            throw new IllegalArgumentException("Unknown fractal: " + fractalName);
        }
        FractalRenderingModule module = moduleInstances.get(fractalName);
        if(!module.isInitialized()){
            module.initialize();
        }

        activeRenderer = new CudaFractalRenderer(module);
        return activeRenderer;
    }

    /**
     * @throws cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaInitializationException
     */
    private static FractalRenderingModule createInstance(Class<? extends FractalRenderingModule> clazz) {
        assert CudaHelpers.isCudaContextThread();
        FractalRenderingModule instance;
        if(Arrays.stream(clazz.getConstructors()).noneMatch(c -> c.getParameterCount() == 0)){
            throw new CudaInitializationException(clazz + " has no parameterless constructor.");
        }
        try {
            instance = clazz.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            throw new CudaInitializationException(e);
        } catch (UnsatisfiedLinkError e) {
            if (e.getMessage().contains("Cuda")) {
                String message = "Error while loading the Cuda native library. Do you have CUDA installed?";
                throw new CudaInitializationException(message, e);
            } else {
                throw new CudaInitializationException(e);
            }
        }
        return instance;
    }
}
