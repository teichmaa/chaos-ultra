package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

public abstract class RenderingKernel extends CudaKernel {

    //final short PARAM_IDX_SURFACE_OUT = 0;
    final short PARAM_IDX_2DARR_OUT;
    final short PARAM_IDX_2DARR_OUT_PITCH;
    private final short PARAM_IDX_WIDTH;
    private final short PARAM_IDX_HEIGHT;
    private final short PARAM_IDX_LEFT_BOTTOM_X;
    private final short PARAM_IDX_LEFT_BOTTOM_Y;
    private final short PARAM_IDX_RIGHT_TOP_X;
    private final short PARAM_IDX_RIGHT_TOP_Y;
    private final short PARAM_IDX_DWELL;

    RenderingKernel(String functionName, CUmodule ownerModule) {
        super(functionName, ownerModule);

        //initialize params[]:
        PARAM_IDX_2DARR_OUT = registerParam();
        PARAM_IDX_2DARR_OUT_PITCH = registerParam();
        PARAM_IDX_WIDTH = registerParam();
        PARAM_IDX_HEIGHT = registerParam();
        PARAM_IDX_LEFT_BOTTOM_X = registerParam();
        PARAM_IDX_LEFT_BOTTOM_Y = registerParam();
        PARAM_IDX_RIGHT_TOP_X = registerParam();
        PARAM_IDX_RIGHT_TOP_Y  = registerParam();
        PARAM_IDX_DWELL = registerParam();

        setWidth(width);
        setHeight(height);
        setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        setDwell(100);
    }

    private int dwell;
    private int width;
    private int height;
    private double left_bottom_x;
    private double left_bottom_y;
    private double right_top_x;
    private double right_top_y;

    /**
     * depending on which float precision the kernel uses (single or double), create cuda typed pointer to given value
     * @param value value to point to (will be truncated by float kernels)
     * @return Float pointer to {@code (float)value} or double pointer to {@code value}
     */
    abstract Pointer pointerToAbstractReal(double value);

    int getDwell() {
        return dwell;
    }

    void setDwell(int dwell) {
        this.dwell = dwell;
        params[PARAM_IDX_DWELL] = pointerTo(dwell);
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    void setWidth(int width) {
        this.width = width;
        params[PARAM_IDX_WIDTH] = pointerTo(width);
    }

    void setHeight(int height) {
        this.height = height;
        params[PARAM_IDX_HEIGHT] = pointerTo(height);
    }

    double getLeft_bottom_x() {
        return left_bottom_x;
    }

    void setLeft_bottom_x(double left_bottom_x) {
        this.left_bottom_x = left_bottom_x;
        params[PARAM_IDX_LEFT_BOTTOM_X] = pointerToAbstractReal(left_bottom_x);
    }

    double getLeft_bottom_y() {
        return left_bottom_y;
    }

    void setLeft_bottom_y(double left_bottom_y) {
        this.left_bottom_y = left_bottom_y;
        params[PARAM_IDX_LEFT_BOTTOM_Y] = pointerToAbstractReal(left_bottom_y);
    }

    double getRight_top_x() {
        return right_top_x;
    }

    void setRight_top_x(double right_top_x) {
        this.right_top_x = right_top_x;
        params[PARAM_IDX_RIGHT_TOP_X] = pointerToAbstractReal(right_top_x);
    }

    double getRight_top_y() {
        return right_top_y;
    }

    void setRight_top_y(double right_top_y) {
        this.right_top_y = right_top_y;
        params[PARAM_IDX_RIGHT_TOP_Y] = pointerToAbstractReal(right_top_y);
    }

    void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {
        setLeft_bottom_x(left_bottom_x);
        setLeft_bottom_y(left_bottom_y);
        setRight_top_x(right_top_x);
        setRight_top_y(right_top_y);
        if(isBoundsAtDoubleLimit()){
            System.out.println("warning: double limit");
        }
    }

    boolean isBoundsAtFloatLimit() {
        double maxAllowedDxError = Math.ulp((float)left_bottom_x);
        double maxAllowedDyError = Math.ulp((float)left_bottom_y);
        double pixelWidth = Math.abs(right_top_x - left_bottom_x) / width;
        double pixelHeight = Math.abs(right_top_y - left_bottom_y) / height;
        return (pixelWidth < maxAllowedDxError) ||( pixelHeight < maxAllowedDyError);
    }
    boolean isBoundsAtDoubleLimit() {
        double maxAllowedDxError = Math.ulp(left_bottom_x);
        double maxAllowedDyError = Math.ulp(left_bottom_y);
        double pixelWidth = Math.abs(right_top_x - left_bottom_x) / width;
        double pixelHeight = Math.abs(right_top_y - left_bottom_y) / height;
        return (pixelWidth < maxAllowedDxError) ||( pixelHeight < maxAllowedDyError);
    }

}
