package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.DefaultFractalModel;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointDoubleReadable;
import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaMemcpyKind;

import java.io.Closeable;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static jcuda.driver.JCudaDriver.cuModuleGetGlobal;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.runtime.JCuda.cudaMemcpy;

/**
 * A structure of JCuda classes, together representing a CUDA module (=1 ptx file) used by CudaFractalRenderer class.
 * Lifecycle:  Closed      --- initialize() --->   Initialized
 *             Initialized  --- close() --->       Closed
 */
public abstract class FractalRenderingModule implements Closeable {

    static final String kernelMainFloat = KernelMainFloat.name;
    static final String kernelMainDouble = KernelMainDouble.name;
    static final String kernelUnderSampled = KernelUnderSampled.name;
    static final String kernelCompose = KernelCompose.name;

    private static final String CUDA_KERNELS_DIR_PROPERTY_NAME = "cudaKernelsDir";
    private static final String CUDA_KERNELS_DIR_PROPERTY_DEFAULT_VALUE = "cudaKernels";
    private static final String PATH_PREFIX;
    private static final String PATH_SUFFIX = ".ptx";

    static {
        //This has to be called from the AWT GL-Thread
        if(!CudaHelpers.isCudaContextThread()){
            throw new IllegalStateException("Cuda has to be initialized from a specific thread.");
        }
        CudaHelpers.cudaInit();

        //locate directory with kernels:
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

//        initialize();
    }

    /**
     * @throws IllegalArgumentException if file with cuda module has not been found
     * @throws CudaInitializationException when other problem with cuda loading occurs
     */
    public void initialize(){
        if(initialized) throw new IllegalStateException("Module already initialized.");
        try {
            module = new CUmodule();
            cuModuleLoad(module, ptxFileFullPath);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_FILE_NOT_FOUND))) {
                String message = "Invalid ptx file name: " + ptxFileFullPath + System.lineSeparator() +
                        "Have you set property " + CUDA_KERNELS_DIR_PROPERTY_NAME + " properly?";
                throw new IllegalArgumentException(message, e);
            } else if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_INVALID_CONTEXT))) {
                throw new CudaInitializationException("Invalid CUDA context", e);
            } else {
                throw new CudaInitializationException(e);
            }
        }

        //add known kernels:
        kernels.put(KernelMainFloat.class, new KernelMainFloat(module));
        kernels.put(KernelMainDouble.class, new KernelMainDouble(module));
        kernels.put(KernelAdvancedFloat.class, new KernelAdvancedFloat(module));
        kernels.put(KernelAdvancedDouble.class, new KernelAdvancedDouble(module));
        kernels.put(KernelUnderSampled.class, new KernelUnderSampled(module));
        kernels.put(KernelCompose.class, new KernelCompose(module));
        kernels.put(KernelDebug.class, new KernelDebug(module));

        initialized = true;
    }

    public boolean isInitialized() {
        return initialized;
    }

    private boolean initialized = false;
    private final String ptxFileFullPath;
    private final String fractalName;

    private CUmodule module;

    private Map<Class<? extends CudaKernel>, CudaKernel> kernels = new HashMap<>();

    protected String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    public String getFractalName() {
        return fractalName;
    }

    protected CUmodule getModule() {
        if(!initialized) throw new IllegalStateException("Module has not been initialized.");
        return module;
    }

    <T extends CudaKernel> T getKernel(Class<T> kernel) {
        if(!initialized) throw new IllegalStateException("Module has not been initialized or has been closed.");
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
        initialized = false;
    }

    public void setFractalCustomParameters(String params) {

    }

    /**
     *
     * @param nameOfTheConstant name of the constant as declared in the cuda source.
     * @return pointer to the constant, and allocated memory size
     */
    private CuSizedDeviceptr getDeviceConstantMemoryPointer(String nameOfTheConstant){
        //from https://github.com/jcuda/jcuda/issues/17
        CUdeviceptr result = new CUdeviceptr();
        long[] paramSizeArray = {0};
        cuModuleGetGlobal(result, paramSizeArray, getModule(), nameOfTheConstant);
        return CuSizedDeviceptr.of(result, paramSizeArray[0]);
    }

    /**
     *
     * @param nameOfTheConstant name of the constant as declared in the cuda source.
     * @param value value to write
     */
    protected void writeToConstantMemory(String nameOfTheConstant, PointDoubleReadable value){
        CuSizedDeviceptr sizedPtr = getDeviceConstantMemoryPointer(nameOfTheConstant);
        if(sizedPtr.getSize() < 2 * Sizeof.DOUBLE)
            throw new IllegalArgumentException("Attempt to write PointDouble to device memory allocated to size " + sizedPtr.getSize());
        double[] source = new double[]{value.getX(), value.getY()};
        cudaMemcpy(sizedPtr.getPtr(), Pointer.to(source), sizedPtr.getSize(), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    /**
     *
     * @param nameOfTheConstant name of the constant as declared in the cuda source.
     * @param value value to write
     */
    protected void writeToConstantMemory(String nameOfTheConstant, int value){
        CuSizedDeviceptr sizedPtr = getDeviceConstantMemoryPointer(nameOfTheConstant);
        if(sizedPtr.getSize() < Sizeof.INT)
            throw new IllegalArgumentException("Attempt to write double to device memory allocated to size " + sizedPtr.getSize());
        int[] source = new int[]{value};
        cudaMemcpy(sizedPtr.getPtr(), Pointer.to(source), sizedPtr.getSize(), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    /**
     *
     * @param nameOfTheConstant name of the constant as declared in the cuda source.
     * @param value value to write
     */
    protected void writeToConstantMemory(String nameOfTheConstant, double value){
        CuSizedDeviceptr sizedPtr = getDeviceConstantMemoryPointer(nameOfTheConstant);
        if(sizedPtr.getSize() < Sizeof.DOUBLE)
            throw new IllegalArgumentException("Attempt to write double to device memory allocated to size " + sizedPtr.getSize());
        double[] source = new double[]{value};
        cudaMemcpy(sizedPtr.getPtr(), Pointer.to(source), sizedPtr.getSize(), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    protected void writeToConstantMemory(String nameOfTheConstant, double[] data){
        CuSizedDeviceptr sizedPtr = getDeviceConstantMemoryPointer(nameOfTheConstant);
        if(sizedPtr.getSize() < data.length * Sizeof.DOUBLE)
            throw new IllegalArgumentException("Attempt to write double[] of size "+ data.length + " to device memory allocated to size " + sizedPtr.getSize());
        cudaMemcpy(sizedPtr.getPtr(), Pointer.to(data), sizedPtr.getSize(), cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

    /**
     * @param params list of numbers, delimited by comma `,` or semicolon `;`.
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
     * @param params list of integers, delimited by comma `,` or semicolon `;`.
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

    /**
     *
     * @param params Input format: `key=value` pairs, delimited by comma `,` or semicolon `;`.
     *               If pair contains multiple equal signs `=`, the remaining ones are considered value. If pair does not contain any equal sign, it is silently skipped.
     */
    protected HashMap<String, String> parseParamsAsKeyValPairs(String params) {
        HashMap<String, String> result = new HashMap<>();
        String[] tokens = params.split("[,;]");
        for (int i = 0; i < tokens.length; i++) {
            String[] pair = tokens[i].split("=", 2);
            if(pair.length < 2) continue;
            result.put(pair[0].trim(),pair[1].trim());
        }
        return result;
    }

    /**
     * May be overridden by concrete modules if a specific default values are required.
     * @param model model to set values to
     */
    protected void supplyDefaultValues(DefaultFractalModel model){
        /* nothing */
    }
}
