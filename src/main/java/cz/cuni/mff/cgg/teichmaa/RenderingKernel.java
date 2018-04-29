package cz.cuni.mff.cgg.teichmaa;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

/**
 * Represents struct of JCuda classes, together representing a CUDA kernel used by CudaRenderer class.
 * The Main function must satisfy this condition:
 * first parameter is 2D array of integers, used as output ({@code int** outputData})
 * second parameter is array's pitch ({@code (long pitch})
 * those parameters must NOT be returned by {@code getKernelParams} method
 */
public abstract class RenderingKernel {

    public final short PARAM_IDX_DEVICE_OUT = 0;
    public final short PARAM_IDX_PITCH = 1;
    public final short PARAM_IDX_WIDTH = 2;
    public final short PARAM_IDX_HEIGHT = 3;
    public final short PARAM_IDX_LEFT_BOTTOM_X = 4;
    public final short PARAM_IDX_LEFT_BOTTOM_Y = 5;
    public final short PARAM_IDX_RIGHT_TOP_X = 6;
    public final short PARAM_IDX_RIGHT_TOP_Y = 7;


    public RenderingKernel(String ptxFileFullPath, String mainFunctionName, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        this.ptxFileFullPath = ptxFileFullPath;
        this.mainFunctionName = mainFunctionName;
        this.width = width;
        this.height = height;
        this.left_bottom_x = left_bottom_x;
        this.left_bottom_y = left_bottom_y;
        this.right_top_x = right_top_x;
        this.right_top_y = right_top_y;


        params = new ArrayList<>(PARAM_IDX_RIGHT_TOP_Y+1);
        for (int i = 0; i < PARAM_IDX_RIGHT_TOP_Y+1; i++) {
            params.add(null);
        }
        params.set(PARAM_IDX_WIDTH, Pointer.to(new int[]{width}));
        params.set(PARAM_IDX_HEIGHT, Pointer.to(new int[]{height}));
        params.set(PARAM_IDX_LEFT_BOTTOM_X, Pointer.to(new float[]{left_bottom_x}));
        params.set(PARAM_IDX_LEFT_BOTTOM_Y, Pointer.to(new float[]{left_bottom_y}));
        params.set(PARAM_IDX_RIGHT_TOP_X, Pointer.to(new float[]{right_top_x}));
        params.set(PARAM_IDX_RIGHT_TOP_Y, Pointer.to(new float[]{right_top_y}));
    }

    private final String ptxFileFullPath;
    private final String mainFunctionName;
    private final int width;
    private final int height;
    private float left_bottom_x;
    private float left_bottom_y;
    private float right_top_x;
    private float right_top_y;
    private CUmodule module;
    private CUfunction mainFunction;
    private List<NativePointerObject> params;

    public String getPtxFileFullPath() {
        return ptxFileFullPath;
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
        params.set(PARAM_IDX_LEFT_BOTTOM_X, Pointer.to(new float[]{left_bottom_x}));
    }

    public float getLeft_bottom_y() {
        return left_bottom_y;
    }

    public void setLeft_bottom_y(float left_bottom_y) {
        this.left_bottom_y = left_bottom_y;
        params.set(PARAM_IDX_LEFT_BOTTOM_Y, Pointer.to(new float[]{left_bottom_y}));
    }

    public float getRight_top_x() {
        return right_top_x;
    }

    public void setRight_top_x(float right_top_x) {
        this.right_top_x = right_top_x;
        params.set(PARAM_IDX_RIGHT_TOP_X, Pointer.to(new float[]{right_top_x}));
    }

    public float getRight_top_y() {
        return right_top_y;
    }

    public void setRight_top_y(float right_top_y) {
        this.right_top_y = right_top_y;
        params.set(PARAM_IDX_RIGHT_TOP_Y, Pointer.to(new float[]{right_top_y}));
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


    protected List<NativePointerObject> getKernelParamsInternal() {
        return params;
    }

    /**
     * @return Array of Kernel's specific parameters. First two fields are reserved for deviceOut and pitch parameters.
     */
    public abstract NativePointerObject[] getKernelParams();

    @Override
    public String toString() {
        return "RenderingKernel "+mainFunctionName + ", size: " +width +" x " + height;
    }
}
