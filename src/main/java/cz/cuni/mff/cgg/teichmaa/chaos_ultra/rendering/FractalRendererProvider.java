package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaFractalRenderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaInitializationException;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.FractalRenderingModule;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleJulia;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.modules.ModuleMandelbrot;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class FractalRendererProvider {

    private static final HashSet<Class<? extends FractalRenderingModule>> modules = new HashSet<>();
    private static final Map<String, FractalRenderingModule> moduleInstances;

    //register new fractals here:
    static {
        modules.add(ModuleJulia.class);
        modules.add(ModuleMandelbrot.class);

        moduleInstances = modules.stream().map(FractalRendererProvider::createInstance).collect(Collectors.toMap(FractalRenderingModule::getFractalName, Function.identity()));
    }

    public static Set<String> getAvailableFractals() {
        return moduleInstances.keySet();
    }


    public FractalRendererProvider(OpenGLParams glParams, ChaosUltraRenderingParams chaosParams) {
        this.glParams = glParams;
        this.chaosParams = chaosParams;
    }

    private OpenGLParams glParams;
    private ChaosUltraRenderingParams chaosParams;

    //    private CudaFractalRenderer activeRenderer;
    private CudaFractalRenderer mandelbrot;
    private CudaFractalRenderer julia;

    public FractalRenderer getRenderer(String fractalName) {
        if (fractalName.equals("mandelbrot")) {
            if (mandelbrot == null) {
                FractalRenderingModule module = moduleInstances.get(fractalName);
                module.initialize();
                mandelbrot = new CudaFractalRenderer(module, glParams, chaosParams);
            }
            return mandelbrot;
        } else if (fractalName.equals("julia")) {
            if (julia == null) {
                FractalRenderingModule module = moduleInstances.get(fractalName);
                module.initialize();
//                julia = new CudaFractalRenderer(module, glParams, chaosParams, mandelbrot.getOutputTextureResource(), mandelbrot.getOutputTextureResource());
                julia = new CudaFractalRenderer(module, glParams, chaosParams);
            }
            return julia;
        }
        else return null;

//        if(activeRenderer !=null){
//            if(activeRenderer.getFractalName().equals(fractalName))
//                return activeRenderer; //returning the current active renderer
//            else{
//                activeRenderer.close(); //closing the previous renderer
//            }
//        }
//
//        if (!moduleInstances.containsKey(fractalName)) {
//            throw new IllegalArgumentException("Unknown fractal: " + fractalName);
//        }
//        FractalRenderingModule module = moduleInstances.get(fractalName);
//        if(!module.isInitialized()){
//            module.initialize();
//        }
//
//        //TODO tohle je nejaky divny slozity, jak tu je vic rendereru a sdili spolu params. To je potreba jeste domyslet.
//        activeRenderer = new CudaFractalRenderer(module, glParams, chaosParams);
//        return activeRenderer;
    }

    /**
     * @throws cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.CudaInitializationException
     */
    private static FractalRenderingModule createInstance(Class<? extends FractalRenderingModule> c) {
        FractalRenderingModule instance;
        try {
            instance = c.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            throw new CudaInitializationException(e);
        } catch (UnsatisfiedLinkError e) {
            //TODO jak jinak tohle zobrazit uzivateli?
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
