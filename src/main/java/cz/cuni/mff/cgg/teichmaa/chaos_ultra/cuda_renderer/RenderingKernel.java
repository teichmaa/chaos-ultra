package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.Model;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmodule;

public abstract class RenderingKernel extends CudaKernel {

    //final short PARAM_IDX_SURFACE_OUT = 0;
    private final short PARAM_IDX_2D_ARR_OUT;
    private final short PARAM_IDX_2D_ARR_OUT_PITCH;
    private final short PARAM_IDX_OUT_SIZE;
    private final short PARAM_IDX_IMAGE;
    private final short PARAM_IDX_MAX_ITERATIONS;

    RenderingKernel(String functionName, CUmodule ownerModule) {
        super(functionName, ownerModule);

        //initialize heuristicsParams[]:
        PARAM_IDX_2D_ARR_OUT = registerParam();
        PARAM_IDX_2D_ARR_OUT_PITCH = registerParam();
        PARAM_IDX_OUT_SIZE = registerParam();
        PARAM_IDX_IMAGE = registerParam();
        PARAM_IDX_MAX_ITERATIONS = registerParam();

        setOutputSize(width, height);
        setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        setMaxIterations(1);
    }

    @Override
    public void setParamsFromModel(Model model) {
        super.setParamsFromModel(model);
        setMaxIterations(model.getMaxIterations());
        setBounds(
                model.getPlaneSegment().getLeftBottom().getX(),
                model.getPlaneSegment().getLeftBottom().getY(),
                model.getPlaneSegment().getRightTop().getX(),
                model.getPlaneSegment().getRightTop().getY()
        );
    }

    private int maxIterations;
    private int width;
    private int height;
    private double left_bottom_x;
    private double left_bottom_y;
    private double right_top_x;
    private double right_top_y;

    /**
     * depending on which float precision the kernel uses (single or double), create cuda typed pointer to given value
     * @param value value to point to (will be truncated by float kernels)
     * @return Float pointer to {@code (float)value} or double pointer to {@code (double)value}
     */
    abstract Pointer pointerToAbstractReal(double value);

    /**
     * Same behaviour as pointerToAbstractReal(double value)
     */
    abstract Pointer pointerToAbstractReal(double v1,double v2,double v3,double v4);

    int getMaxIterations() {
        return maxIterations;
    }

    void setMaxIterations(int maxIterations) {
        if(maxIterations < 1) throw new IllegalArgumentException("maxIterations must be a positive number, but is : " + maxIterations);
        this.maxIterations = maxIterations;
        params[PARAM_IDX_MAX_ITERATIONS] = CudaHelpers.pointerTo(maxIterations);
    }

    int getWidth() {
        return width;
    }

    int getHeight() {
        return height;
    }

    void setOutputSize(int width, int height) {
        this.width = width;
        this.height = height;
        params[PARAM_IDX_OUT_SIZE] = CudaHelpers.pointerTo(width, height);
    }

    void setOutput(CUdeviceptr output, long outputPitch){
        params[PARAM_IDX_2D_ARR_OUT] = Pointer.to(output);
        params[PARAM_IDX_2D_ARR_OUT_PITCH] = CudaHelpers.pointerTo(outputPitch);
    }

    double getLeft_bottom_x() {
        return left_bottom_x;
    }

    double getLeft_bottom_y() {
        return left_bottom_y;
    }

    double getRight_top_x() {
        return right_top_x;
    }

    double getRight_top_y() {
        return right_top_y;
    }

    void setBounds(double left_bottom_x, double left_bottom_y, double right_top_x, double right_top_y) {
        params[PARAM_IDX_IMAGE] = pointerToAbstractReal(left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        this.left_bottom_x = left_bottom_x;
        this.left_bottom_y = left_bottom_y;
        this.right_top_x = right_top_x;
        this.right_top_y = right_top_y;
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
