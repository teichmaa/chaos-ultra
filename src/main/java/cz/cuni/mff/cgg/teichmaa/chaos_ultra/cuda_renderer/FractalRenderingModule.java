package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.CudaException;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.io.Closeable;
import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * A structure of JCuda classes, together representing a CUDA module (=1 ptx file) used by FractalRenderer class.
 */
abstract class FractalRenderingModule implements Closeable {

    static final String kernelInit = KernelInit.name;
    static final String kernelMainFloat = KernelMainFloat.name;
    static final String kernelMainDouble = KernelMainDouble.name;
    static final String kernelUnderSampled = KernelUnderSampled.name;
    static final String kernelCompose = KernelCompose.name;

    /**
     * using an absolute path is a temporary workaround and should be fixed
     */
    private static final String pathAbsolutePrefix = "E:\\Tonda\\Desktop\\chaos-ultra\\";
    private static final String pathLocalPrefix = "src\\main\\cuda\\";
    private static final String pathSuffix = ".ptx";

    static {
        CudaHelpers.cudaInit();
    }

    /**
     *
     * @param ptxFileName name of the cuda-compiled file containing the module, without '.ptx'
     * @param fractalName name of the fractal that this module represents
     */
    FractalRenderingModule(String ptxFileName, String fractalName) {
        this.ptxFileFullPath = pathAbsolutePrefix + pathLocalPrefix + ptxFileName + pathSuffix;
        this.fractalName = fractalName;

        //module load:
        // Using absolute file path because I cannot make it working with relative paths
        //      (this is because I deploy with maven, and run the app from jar, where the nvcc and other invoked proccesses cannot find the source files )
        try {
            module = new CUmodule();
            cuModuleLoad(module, ptxFileFullPath);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_FILE_NOT_FOUND))) {
                throw new IllegalArgumentException("Invalid ptx file name: " + ptxFileFullPath, e);
            } else if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_INVALID_CONTEXT))) {
                throw new IllegalArgumentException("Invalid CUDA context", e);
            } else{
                throw e;
            }
        }

        //add known kernels:
        kernels.put(KernelInit.class, new KernelInit(module));
        kernels.put(KernelMainFloat.class, new KernelMainFloat(module));
        kernels.put(KernelMainDouble.class, new KernelMainDouble(module));
        kernels.put(KernelAdvancedFloat.class, new KernelAdvancedFloat(module));
        kernels.put(KernelAdvancedDouble.class, new KernelAdvancedDouble(module));
        kernels.put(KernelUnderSampled.class, new KernelUnderSampled(module));
        kernels.put(KernelCompose.class, new KernelCompose(module));
        kernels.put(KernelDebug.class, new KernelDebug(module));

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
        if(module == null) throw new IllegalStateException("the module has been closed.");
        if (!kernels.containsKey(kernel)) {
            throw new IllegalArgumentException("No such kernel available: " + kernel.getCanonicalName());
        }
        return (T) kernels.get(kernel); //this cast will always be successful
    }

    @Override
    public String toString() {
        return "FractalRenderingModule " + fractalName;
    }

    @Override
    public void close() {
        if(module != null){
            JCudaDriver.cuModuleUnload(module);
            module = null;
        }
    }
}
