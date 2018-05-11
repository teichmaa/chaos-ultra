package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.Pointer;

public class JuliaKernel extends FractalRenderingKernel {

    public final short PARAM_IDX_C_X;
    public final short PARAM_IDX_C_Y;

    public JuliaKernel(float cx, float cy) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\julia.ptx", "julia",null);
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
