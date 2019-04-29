package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.CudaException;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.io.Closeable;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * A structure of JCuda classes, together representing a CUDA module (=1 ptx file) used by CudaFractalRenderer class.
 */
abstract class FractalRenderingModule implements Closeable {

    static final String kernelInit = KernelInit.name;
    static final String kernelMainFloat = KernelMainFloat.name;
    static final String kernelMainDouble = KernelMainDouble.name;
    static final String kernelUnderSampled = KernelUnderSampled.name;
    static final String kernelCompose = KernelCompose.name;

    private static final String CUDA_KERNELS_DIR_PROPERTY_NAME = "cudaKernelsDir";
    private static final String CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE = "cudaKernels";
    private static final String PATH_PREFIX;
    private static final String PATH_SUFFIX = ".ptx";

    static {
        CudaHelpers.cudaInit();

        if(System.getProperty(CUDA_KERNELS_DIR_PROPERTY_NAME) == null){
            System.setProperty(CUDA_KERNELS_DIR_PROPERTY_NAME, CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE);
            System.err.println("cudaKernelsDir property not specified, fallbacking to '" + CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE + "' (try starting java with -DcudaKernelsDir=<directory relative loaction>");
        }
        PATH_PREFIX =  System.getProperty("user.dir") + File.separator + System.getProperty(CUDA_KERNELS_DIR_PROPERTY_NAME);
    }

    /**
     *
     * @param ptxFileName name of the cuda-compiled file containing the module, without '.ptx'
     * @param fractalName name of the rendering that this module represents
     */
    FractalRenderingModule(String ptxFileName, String fractalName) {
        this.ptxFileFullPath = PATH_PREFIX + File.separator + ptxFileName + PATH_SUFFIX;
        this.fractalName = fractalName;

        //module load:
        try {
            module = new CUmodule();
            cuModuleLoad(module, ptxFileFullPath);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_FILE_NOT_FOUND))) {
                String message = "Invalid ptx file name: " + ptxFileFullPath + System.lineSeparator() +
                        "Have you set " + CUDA_KERNELS_DIR_PROPERTY_NAME + " properly?";
                throw new IllegalArgumentException(message, e);
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
            throw new IllegalArgumentException("No such kernel available: " + kernel.getSimpleName());
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
