package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * Represents struct of JCuda classes, together representing a CUDA kernel used by CudaLauncher class.
 * The Main function must satisfy this condition:
 * first parameter is 2D array of integers, used as output ({@code int** outputData})
 * second parameter is array's pitch ({@code (long pitch})
 * those parameters must NOT be returned by {@code getKernelParams} method
 */
public abstract class AbstractFractalRenderKernel {

    public final short PARAM_IDX_DEVICE_OUT = 0;
    public final short PARAM_IDX_PITCH = 1;
    public final short PARAM_IDX_WIDTH = 2;
    public final short PARAM_IDX_HEIGHT = 3;
    public final short PARAM_IDX_LEFT_BOTTOM_X = 4;
    public final short PARAM_IDX_LEFT_BOTTOM_Y = 5;
    public final short PARAM_IDX_RIGHT_TOP_X = 6;
    public final short PARAM_IDX_RIGHT_TOP_Y = 7;
    public final short PARAM_IDX_DWELL = 8;


    public AbstractFractalRenderKernel(String ptxFileFullPath, String mainFunctionName, int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        this.ptxFileFullPath = ptxFileFullPath;
        this.mainFunctionName = mainFunctionName;
        this.dwell = dwell;
        this.width = width;
        this.height = height;
        this.left_bottom_x = left_bottom_x;
        this.left_bottom_y = left_bottom_y;
        this.right_top_x = right_top_x;
        this.right_top_y = right_top_y;

        params = new NativePointerObject[PARAM_IDX_DWELL+1];

        params[ PARAM_IDX_WIDTH] = Pointer.to(new int[]{width});
        params[ PARAM_IDX_HEIGHT] = Pointer.to(new int[]{height});
        params[ PARAM_IDX_LEFT_BOTTOM_X] = Pointer.to(new float[]{left_bottom_x});
        params[ PARAM_IDX_LEFT_BOTTOM_Y] = Pointer.to(new float[]{left_bottom_y});
        params[ PARAM_IDX_RIGHT_TOP_X] = Pointer.to(new float[]{right_top_x});
        params[ PARAM_IDX_RIGHT_TOP_Y] = Pointer.to(new float[]{right_top_y});
        params[ PARAM_IDX_DWELL] = Pointer.to(new int[]{dwell});
    }

    /**
     *
     * @param p kernel parameter to add
     * @return index of the added param
     */
    protected short addParam(NativePointerObject p){
        NativePointerObject[] newParams = new NativePointerObject[params.length + 1];
        for (int i = 0; i < params.length; i++) {
            newParams[i] = params[i];
        }
        int idx = newParams.length-1;
        newParams[idx] = p;
        params = newParams;
        return (short) idx;
    }

    private final String ptxFileFullPath;
    private final String mainFunctionName;
    private int dwell;
    private final int width;
    private final int height;
    private float left_bottom_x;
    private float left_bottom_y;
    private float right_top_x;
    private float right_top_y;
    private CUmodule module;
    private CUfunction mainFunction;
    private NativePointerObject[] params;

    public String getPtxFileFullPath() {
        return ptxFileFullPath;
    }

    public int getDwell() {
        return dwell;
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


//    protected List<NativePointerObject> getKernelParamsInternal() {
//        return params;
//    }

    /**
     * @return Array of Kernel's specific parameters. First two fields are reserved for deviceOut and pitch parameters.
     */
    public NativePointerObject[] getKernelParams(){return params;}

    @Override
    public String toString() {
        return "AbstractFractalRenderKernel " + mainFunctionName + ", size: " + width + " x " + height + ", dwell: " + dwell;
    }
}