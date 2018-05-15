package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * A structure of JCuda classes, together representing a CUDA kernel used by FractalRenderer class.
 * The Main function must satisfy this condition:
 * * First parameter is 2D array of integers, used as output ({@code int** outputData}).
 * * Second parameter is array's pitch ({@code (long pitch}).
 * Those parameters will be post-defined by the caller of {@code getKernelParams} method.
 */
abstract class FractalRenderingKernel {

    final short PARAM_IDX_SURFACE_OUT = 0;
    final short PARAM_IDX_PITCH = 1;
    private final short PARAM_IDX_WIDTH = 2;
    private final short PARAM_IDX_HEIGHT = 3;
    private final short PARAM_IDX_LEFT_BOTTOM_X = 4;
    private final short PARAM_IDX_LEFT_BOTTOM_Y = 5;
    private final short PARAM_IDX_RIGHT_TOP_X = 6;
    private final short PARAM_IDX_RIGHT_TOP_Y = 7;
    private final short PARAM_IDX_DWELL = 8;
    final short PARAM_IDX_DEVICE_OUT = 9;
    final short PARAM_IDX_SURFACE_PALETTE = 10;
    final short PARAM_IDX_PALETTE_LENGTH = 11;
    final short PARAM_IDX_RANDOM_SAMPLES = 12;
    private final short PARAM_IDX_SUPER_SAMPLING_LEVEL = 13;
    private final short PARAM_IDX_ADAPTIVE_SS = 14;
    private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS = 15;


    /**
     * @param ptxFileFullPath
     * @param mainFunctionName
     * @param initFunctionName name of the kernel function to be called before start. If null or empty, no function will be called to init.
     */
    FractalRenderingKernel(String ptxFileFullPath, String mainFunctionName, String initFunctionName) {
        this.ptxFileFullPath = ptxFileFullPath;
        this.mainFunctionName = mainFunctionName;
        this.initFunctionName = initFunctionName;
        hasInitFunction = !(initFunctionName == null || initFunctionName.isEmpty());

        params = new NativePointerObject[PARAM_IDX_VISUALISE_ADAPTIVE_SS + 1];
        //initialize params[] :
        setWidth(width);
        setHeight(height);
        setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        setDwell(100);
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
        setSuperSamplingLevel(1);
    }

    /**
     * @param p kernel parameter to add
     * @return index of the added param
     */
    protected short addParam(NativePointerObject p) {
        NativePointerObject[] newParams = new NativePointerObject[params.length + 1];
        for (int i = 0; i < params.length; i++) {
            newParams[i] = params[i];
        }
        int idx = newParams.length - 1;
        newParams[idx] = p;
        params = newParams;
        return (short) idx;
    }

    private final String ptxFileFullPath;
    private final String mainFunctionName;
    private final String initFunctionName;
    private final boolean hasInitFunction;
    private int dwell;
    private int width;
    private int height;
    private float left_bottom_x;
    private float left_bottom_y;
    private float right_top_x;
    private float right_top_y;
    private int superSamplingLevel;
    private boolean adaptiveSS;
    private boolean visualiseAdaptiveSS;
    private CUmodule module;
    private CUfunction mainFunction;
    private CUfunction initFunction;
    private NativePointerObject[] params;

    String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    String getInitFunctionName() {
        return initFunctionName;
    }

    boolean isInitiable() {
        return hasInitFunction;
    }

    boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = Pointer.to(new int[]{visualiseAdaptiveSS ? 1 : 0});
    }

    boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = Pointer.to(new int[]{adaptiveSS ? 1 : 0});
    }

    int getDwell() {
        return dwell;
    }

    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = Pointer.to(new int[]{superSamplingLevel});
    }

    int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

    void setDwell(int dwell) {
        this.dwell = dwell;
        params[PARAM_IDX_DWELL] = Pointer.to(new int[]{dwell});
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    void setWidth(int width) {
        this.width = width;
        params[PARAM_IDX_WIDTH] = Pointer.to(new int[]{width});
    }

    void setHeight(int height) {
        this.height = height;
        params[PARAM_IDX_HEIGHT] = Pointer.to(new int[]{height});
    }

    float getLeft_bottom_x() {
        return left_bottom_x;
    }

    void setLeft_bottom_x(float left_bottom_x) {
        this.left_bottom_x = left_bottom_x;
        params[PARAM_IDX_LEFT_BOTTOM_X] = Pointer.to(new float[]{left_bottom_x});
    }

    float getLeft_bottom_y() {
        return left_bottom_y;
    }

    void setLeft_bottom_y(float left_bottom_y) {
        this.left_bottom_y = left_bottom_y;
        params[PARAM_IDX_LEFT_BOTTOM_Y] = Pointer.to(new float[]{left_bottom_y});
    }

    float getRight_top_x() {
        return right_top_x;
    }

    void setRight_top_x(float right_top_x) {
        this.right_top_x = right_top_x;
        params[PARAM_IDX_RIGHT_TOP_X] = Pointer.to(new float[]{right_top_x});
    }

    float getRight_top_y() {
        return right_top_y;
    }

    void setRight_top_y(float right_top_y) {
        this.right_top_y = right_top_y;
        params[PARAM_IDX_RIGHT_TOP_Y] = Pointer.to(new float[]{right_top_y});
    }

    void setBounds(float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        setLeft_bottom_x(left_bottom_x);
        setLeft_bottom_y(left_bottom_y);
        setRight_top_x(right_top_x);
        setRight_top_y(right_top_y);
        if (isNumbersTooSmall())
            System.err.println("Warning: rendering kernel bounds are too near to each other");
    }

    boolean isNumbersTooSmall() {
        float EPSILON = Math.ulp(1f);
        return (Math.abs(right_top_x - left_bottom_x) / width < EPSILON)
                || (Math.abs(right_top_y - left_bottom_y) / height < EPSILON);
    }

    String getMainFunctionName() {
        return mainFunctionName;
    }


    CUmodule getModule() {
        // Using absolute file path because I cannot make it working with relative paths
        //      (this is because I deploy with maven, and run the app from jar, where the nvcc and other invoked proccesses cannot find the source files )
        if (module == null) {
            module = new CUmodule();
            try {
                cuModuleLoad(module, ptxFileFullPath);
            } catch (CudaException e) {
                if (e.getMessage().contains(CUresult.stringFor(CUresult.CUDA_ERROR_FILE_NOT_FOUND))) {
                    System.err.println("Invalid ptx file name: " + ptxFileFullPath);
                } else {
                    throw e;
                }

            }
        }
        return module;
    }

    CUfunction getMainFunction() {
        if (mainFunction == null) {
            mainFunction = new CUfunction();
            cuModuleGetFunction(mainFunction, getModule(), mainFunctionName);
        }
        return mainFunction;
    }

    CUfunction getInitFunction() {
        if (!hasInitFunction)
            throw new UnsupportedOperationException("cannot call getInitFunction on a kernel without init function");
        if (initFunction == null) {
            initFunction = new CUfunction();
            cuModuleGetFunction(initFunction, getModule(), initFunctionName);
        }
        return initFunction;
    }

    /**
     * @return Array of Kernel's specific parameters. Fields to which this class presents a public index must be defined by the caller.
     */
    NativePointerObject[] getKernelParams() {
        return params;
    }

    @Override
    public String toString() {
        return "FractalRenderingKernel " + mainFunctionName + ", size: " + width + " x " + height + ", dwell: " + dwell;
    }
}
