package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUmodule;

public abstract class RenderingKernel extends CudaKernel {

    final short PARAM_IDX_SURFACE_OUT = 0;
    private final short PARAM_IDX_WIDTH = 1;
    private final short PARAM_IDX_HEIGHT = 2;
    final short PARAM_IDX_SURFACE_PALETTE = 3;
    final short PARAM_IDX_PALETTE_LENGTH = 4;
    private final short PARAM_IDX_LEFT_BOTTOM_X = 5;
    private final short PARAM_IDX_LEFT_BOTTOM_Y = 6;
    private final short PARAM_IDX_RIGHT_TOP_X = 7;
    private final short PARAM_IDX_RIGHT_TOP_Y = 8;
    private final short PARAM_IDX_DWELL = 9;

    RenderingKernel(String functionName, CUmodule ownerModule) {
        super(functionName, ownerModule);

        params = new NativePointerObject[PARAM_IDX_DWELL + 1];
        //initialize params[] :
        setWidth(width);
        setHeight(height);
        setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        setDwell(100);
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

    private int dwell;
    private int width;
    private int height;
    private float left_bottom_x;
    private float left_bottom_y;
    private float right_top_x;
    private float right_top_y;

    protected NativePointerObject[] params;

    int getDwell() {
        return dwell;
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

    @Override
    public NativePointerObject[] getKernelParams() {
        return params;
    }

}
