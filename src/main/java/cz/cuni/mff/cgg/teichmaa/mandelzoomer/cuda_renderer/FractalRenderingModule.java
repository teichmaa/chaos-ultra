package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.CudaException;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;

import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * A structure of JCuda classes, together representing a CUDA module (=1 ptx file) used by FractalRenderer class.
 */
abstract class FractalRenderingModule {

    static final String functionInit = KernelInit.name;
    static final String functionMain = KernelFractalRenderMain.name;
    static final String functionSupSampled = "fractalRenderSupSampled";

    static {
        CudaHelpers.cudaInit();
    }

    FractalRenderingModule(String ptxFileFullPath, String fractalName) {
        this.ptxFileFullPath = ptxFileFullPath;
        this.fractalName = fractalName;

        //module load:
        // Using absolute file path because I cannot make it working with relative paths
        //      (this is because I deploy with maven, and run the app from jar, where the nvcc and other invoked proccesses cannot find the source files )
        try {
            module = new CUmodule();
            cuModuleLoad(module, ptxFileFullPath);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_FILE_NOT_FOUND))) {
                System.err.println("Invalid ptx file name: " + ptxFileFullPath);
            } else {
                throw e;
            }
        }

        //add known kernels:
        kernels.put(KernelFractalRenderMain.class, new KernelFractalRenderMain(module));
        kernels.put(KernelInit.class, new KernelInit(module));
    }

    private final String ptxFileFullPath;
    private final String fractalName;

    private CUmodule module;

    private Map<Class<? extends CudaKernel>, CudaKernel> kernels = new HashMap<>();

    String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    String getFractalName() {
        return fractalName;
    }

    private CUmodule getModule() {
        return module;
    }

    <T extends CudaKernel> T getKernel(Class<T> kernel) {
        if (!kernels.containsKey(kernel)) {
            throw new IllegalArgumentException("No such kernel available: " + kernel.getCanonicalName());
        }
        return (T) kernels.get(kernel); //this cast will always be successful
    }

    @Override
    public String toString() {
        return "FractalRenderingModule " + fractalName;
    }
}
