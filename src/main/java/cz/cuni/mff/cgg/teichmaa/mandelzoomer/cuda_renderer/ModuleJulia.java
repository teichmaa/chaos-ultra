package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.Pointer;

public class ModuleJulia extends FractalRenderingModule {

    public final short PARAM_IDX_C_X;
    public final short PARAM_IDX_C_Y;

    public ModuleJulia(double cx, double cy) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\julia.ptx", "julia");
        this.cx = cx;
        this.cy = cy;

        KernelMain main = super.getKernel(KernelMain.class);

        PARAM_IDX_C_X = main.registerParam((float)cx);
        PARAM_IDX_C_Y = main.registerParam((float)cy);

    }

    private final double cx;
    private final double cy;

    public double getCx() {
        return cx;
    }

    public double getCy() {
        return cy;
    }


}
