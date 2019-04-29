package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.*;
import jcuda.CudaException;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.io.Closeable;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * A structure of JCuda classes, together representing a CUDA module (=1 ptx file) used by CudaFractalRenderer class.
 */
public abstract class FractalRenderingModule implements Closeable {

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

        if (System.getProperty(CUDA_KERNELS_DIR_PROPERTY_NAME) == null) {
            System.setProperty(CUDA_KERNELS_DIR_PROPERTY_NAME, CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE);
            System.err.println(CUDA_KERNELS_DIR_PROPERTY_NAME + " property not specified, fallbacking to '" + CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE + "' (try starting java with -D" + CUDA_KERNELS_DIR_PROPERTY_NAME + "=<directory relative location>");
        }
        PATH_PREFIX = System.getProperty("user.dir") + File.separator + System.getProperty(CUDA_KERNELS_DIR_PROPERTY_NAME);
    }

    /**
     * @param ptxFileName name of the cuda-compiled file containing the module, without '.ptx' suffix
     * @param fractalName name of the rendering that this module represents
     */
    protected FractalRenderingModule(String ptxFileName, String fractalName) {
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
            } else {
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

    protected String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    protected String getFractalName() {
        return fractalName;
    }

    protected CUmodule getModule() {
        return module;
    }

    <T extends CudaKernel> T getKernel(Class<T> kernel) {
        if (module == null) throw new IllegalStateException("the module has been closed.");
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
        if (module != null) {
            JCudaDriver.cuModuleUnload(module);
            module = null;
        }
    }

    public abstract void setFractalCustomParameters(String params);

    /**
     * @throws NumberFormatException
     */
    protected double[] parseParamsAsDoubles(String params) {
        String[] tokens = params.split("[,;]");
        List<Double> vals = Arrays.stream(tokens).map(Double::parseDouble).collect(Collectors.toList());
        double[] result = new double[vals.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = vals.get(i);
        }
        return result;
    }

    /**
     * @throws NumberFormatException
     */

    protected int[] parseParamsAsIntegers(String params) {
        String[] tokens = params.split("[,;]");
        List<Integer> vals = Arrays.stream(tokens).map(Integer::parseInt).collect(Collectors.toList());
        int[] result = new int[vals.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = vals.get(i);
        }
        return result;
    }

    protected HashMap<String, String> parseParamsAsKeyValPairs(String params) {
        return null;
        //TODO impl
    }
}
