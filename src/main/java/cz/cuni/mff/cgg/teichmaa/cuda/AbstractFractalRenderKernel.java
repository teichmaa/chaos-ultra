package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * Represents struct of JCuda classes, together representing a CUDA kernel used by CudaLauncher class.
 * The Main function must satisfy this condition:
 * * First parameter is 2D array of integers, used as output ({@code int** outputData}).
 * * Second parameter is array's pitch ({@code (long pitch}).
 * Those parameters will be post-defined by the caller of {@code getKernelParams} method.
 */
public abstract class AbstractFractalRenderKernel {

    public final short PARAM_IDX_SURFACE_OUT = 0;
    public final short PARAM_IDX_PITCH = 1;
    public final short PARAM_IDX_WIDTH = 2;
    public final short PARAM_IDX_HEIGHT = 3;
    public final short PARAM_IDX_LEFT_BOTTOM_X = 4;
    public final short PARAM_IDX_LEFT_BOTTOM_Y = 5;
    public final short PARAM_IDX_RIGHT_TOP_X = 6;
    public final short PARAM_IDX_RIGHT_TOP_Y = 7;
    public final short PARAM_IDX_DWELL = 8;
    public final short PARAM_IDX_DEVICE_OUT = 9;
    public final short PARAM_IDX_SURFACE_PALETTE = 10;
    public final short PARAM_IDX_PALETTE_LENGTH = 11;
    public final short PARAM_IDX_RANDOM_SAMPLES = 12;
    public final short PARAM_IDX_SUPER_SAMPLING_LEVEL = 13;
    public final short PARAM_IDX_ADAPTIVE_SS = 14;
    public final short PARAM_IDX_VISUALISE_ADAPTIVE_SS = 15;


    /**
     *
     * @param ptxFileFullPath
     * @param mainFunctionName
     * @param initFunctionName name of the kernel function to be called before start. If null or empty, no function will be called to init.
     * @param dwell
     * @param width
     * @param height
     * @param left_bottom_x
     * @param left_bottom_y
     * @param right_top_x
     * @param right_top_y
     */
    public AbstractFractalRenderKernel(String ptxFileFullPath, String mainFunctionName, String initFunctionName, int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        this.ptxFileFullPath = ptxFileFullPath;
        this.mainFunctionName = mainFunctionName;
        this.initFunctionName = initFunctionName;
        hasInitFunction = !(initFunctionName == null || initFunctionName.isEmpty());
        if(!hasInitFunction) initFunctionName = "";
        this.dwell = dwell;
        this.width = width;
        this.height = height;
        this.left_bottom_x = left_bottom_x;
        this.left_bottom_y = left_bottom_y;
        this.right_top_x = right_top_x;
        this.right_top_y = right_top_y;
        superSamplingLevel = 1;

        params = new NativePointerObject[PARAM_IDX_VISUALISE_ADAPTIVE_SS + 1];

        params[PARAM_IDX_WIDTH] = Pointer.to(new int[]{width});
        params[PARAM_IDX_HEIGHT] = Pointer.to(new int[]{height});
        params[PARAM_IDX_LEFT_BOTTOM_X] = Pointer.to(new float[]{left_bottom_x});
        params[PARAM_IDX_LEFT_BOTTOM_Y] = Pointer.to(new float[]{left_bottom_y});
        params[PARAM_IDX_RIGHT_TOP_X] = Pointer.to(new float[]{right_top_x});
        params[PARAM_IDX_RIGHT_TOP_Y] = Pointer.to(new float[]{right_top_y});
        params[PARAM_IDX_DWELL] = Pointer.to(new int[]{dwell});
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = Pointer.to(new int[]{superSamplingLevel});
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
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

    public String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    public String getInitFunctionName() {
        return initFunctionName;
    }

    public boolean isInitiable() {
        return hasInitFunction;
    }

    public boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    public void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = Pointer.to(new int[]{visualiseAdaptiveSS ? 1 : 0});
    }

    public boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    public void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = Pointer.to(new int[]{adaptiveSS ? 1 : 0});
    }

    public int getDwell() {
        return dwell;
    }

    public void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = Pointer.to(new int[]{superSamplingLevel});
    }
    public int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

    public void setDwell(int dwell) {
        this.dwell = dwell;
        params[PARAM_IDX_DWELL] = Pointer.to(new int[]{dwell});
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public void setWidth(int width) {
        this.width = width;
        params[PARAM_IDX_WIDTH] = Pointer.to(new int[]{width});
    }

    public void setHeight(int height) {
        this.height = height;
        params[PARAM_IDX_WIDTH] = Pointer.to(new int[]{height});
    }

    public float getLeft_bottom_x() {
        return left_bottom_x;
    }

    public void setLeft_bottom_x(float left_bottom_x) {
        this.left_bottom_x = left_bottom_x;
        params[PARAM_IDX_LEFT_BOTTOM_X] = Pointer.to(new float[]{left_bottom_x});
    }

    public float getLeft_bottom_y() {
        return left_bottom_y;
    }

    public void setLeft_bottom_y(float left_bottom_y) {
        this.left_bottom_y = left_bottom_y;
        params[PARAM_IDX_LEFT_BOTTOM_Y] = Pointer.to(new float[]{left_bottom_y});
    }

    public float getRight_top_x() {
        return right_top_x;
    }

    public void setRight_top_x(float right_top_x) {
        this.right_top_x = right_top_x;
        params[PARAM_IDX_RIGHT_TOP_X] = Pointer.to(new float[]{right_top_x});
    }

    public float getRight_top_y() {
        return right_top_y;
    }

    public void setRight_top_y(float right_top_y) {
        this.right_top_y = right_top_y;
        params[PARAM_IDX_RIGHT_TOP_Y] = Pointer.to(new float[]{right_top_y});
    }

    public void setBounds(float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        setLeft_bottom_x(left_bottom_x);
        setLeft_bottom_y(left_bottom_y);
        setRight_top_x(right_top_x);
        setRight_top_y(right_top_y);
    }

    public String getMainFunctionName() {
        return mainFunctionName;
    }


    public CUmodule getModule() {
        // Using absolute file path because I cannot make it working with relative paths
        //      (this is because I deploy with maven, and run the app from jar, where the nvcc and other invoked proccesses cannot find the source files )
        if (module == null) {
            module = new CUmodule();
            cuModuleLoad(module, ptxFileFullPath);
        }
        return module;
    }

    public CUfunction getMainFunction() {
        if (mainFunction == null) {
            mainFunction = new CUfunction();
            cuModuleGetFunction(mainFunction, getModule(), mainFunctionName);
        }
        return mainFunction;
    }
    public CUfunction getInitFunction() {
        if(!hasInitFunction)
            throw new UnsupportedOperationException("cannot call getInitFunction on a kernel without init function");
        if (initFunction == null) {
            initFunction = new CUfunction();
            cuModuleGetFunction(initFunction, getModule(), initFunctionName);
        }
        return initFunction;
    }


//    protected List<NativePointerObject> getKernelParamsInternal() {
//        return params;
//    }

    /**
     * @return Array of Kernel's specific parameters. First two fields are reserved for deviceOut and pitch parameters.
     */
    public NativePointerObject[] getKernelParams() {
        return params;
    }

    @Override
    public String toString() {
        return "AbstractFractalRenderKernel " + mainFunctionName + ", size: " + width + " x " + height + ", dwell: " + dwell;
    }
}
