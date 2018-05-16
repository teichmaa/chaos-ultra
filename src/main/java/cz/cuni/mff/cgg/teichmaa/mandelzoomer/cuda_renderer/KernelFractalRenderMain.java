package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer.CudaKernel;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUmodule;

class KernelFractalRenderMain extends CudaKernel {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderMain";

    static final short PARAM_IDX_SURFACE_OUT = 0;
    static final short PARAM_IDX_PITCH = 1;
    static private final short PARAM_IDX_WIDTH = 2;
    static private final short PARAM_IDX_HEIGHT = 3;
    static private final short PARAM_IDX_LEFT_BOTTOM_X = 4;
    static private final short PARAM_IDX_LEFT_BOTTOM_Y = 5;
    static private final short PARAM_IDX_RIGHT_TOP_X = 6;
    static private final short PARAM_IDX_RIGHT_TOP_Y = 7;
    static private final short PARAM_IDX_DWELL = 8;
    static final short PARAM_IDX_DEVICE_OUT = 9;
    static final short PARAM_IDX_SURFACE_PALETTE = 10;
    static final short PARAM_IDX_PALETTE_LENGTH = 11;
    static final short PARAM_IDX_RANDOM_SAMPLES = 12;
    static private final short PARAM_IDX_SUPER_SAMPLING_LEVEL = 13;
    static private final short PARAM_IDX_ADAPTIVE_SS = 14;
    static private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS = 15;

    KernelFractalRenderMain(CUmodule ownerModule) {
        super(name, ownerModule);
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

    private NativePointerObject[] params;

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


    /**
     * @param p kernel parameter to add
     * @return index of the added param
     */
    public short addParam(NativePointerObject p) {
        NativePointerObject[] newParams = new NativePointerObject[params.length + 1];
        for (int i = 0; i < params.length; i++) {
            newParams[i] = params[i];
        }
        int idx = newParams.length - 1;
        newParams[idx] = p;
        params = newParams;
        return (short) idx;
    }


    @Override
    public NativePointerObject[] getKernelParams() {
        return params;
    }
}
