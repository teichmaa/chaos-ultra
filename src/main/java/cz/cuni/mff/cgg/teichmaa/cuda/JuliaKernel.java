package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.Pointer;

public class JuliaKernel extends AbstractFractalRenderKernel {

    public final short PARAM_IDX_C_X;
    public final short PARAM_IDX_C_Y;

    public JuliaKernel(int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, float cx, float cy) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\julia.ptx", "julia",null, dwell, width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        this.cx = cx;
        this.cy = cy;

        PARAM_IDX_C_X = addParam(Pointer.to(new float[]{cy}));
        PARAM_IDX_C_Y = addParam(Pointer.to(new float[]{cx}));

    }

    private final float cx;
    private final float cy;

    public float getCx() {
        return cx;
    }

    public float getCy() {
        return cy;
    }


}
