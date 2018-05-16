package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.Pointer;

public class ModuleJulia extends FractalRenderingModule {

    public final short PARAM_IDX_C_X;
    public final short PARAM_IDX_C_Y;

    public ModuleJulia(float cx, float cy) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\julia.ptx", "julia");
        this.cx = cx;
        this.cy = cy;

        KernelFractalRenderMain main = super.getKernel(KernelFractalRenderMain.class);

        PARAM_IDX_C_X = main.addParam(Pointer.to(new float[]{cy}));
        PARAM_IDX_C_Y = main.addParam(Pointer.to(new float[]{cx}));
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
