package cz.cuni.mff.cgg.teichmaa;

import jcuda.NativePointerObject;
import jcuda.Pointer;

import java.util.List;

public class JuliaKernel extends RenderingKernel {

    public final short PARAM_IDX_DWELL;
    public final short PARAM_IDX_C_X;
    public final short PARAM_IDX_C_Y;

    public JuliaKernel(int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y, float cx, float cy) {
        super("E:\\Tonda\\OneDrive\\baka\\jcuda\\j-fractal-renderer\\src\\main\\resources\\julia.ptx", "julia", width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        this.dwell = dwell;
        this.cx = cx;
        this.cy = cy;

        List<NativePointerObject> ancestorsParams = super.getKernelParamsInternal();
        PARAM_IDX_DWELL = (short) ancestorsParams.size();
        PARAM_IDX_C_X = (short) (PARAM_IDX_DWELL + 1);
        PARAM_IDX_C_Y = (short) (PARAM_IDX_DWELL + 2);
        ancestorsParams.add(Pointer.to(new int[]{dwell}));
        ancestorsParams.add(Pointer.to(new float[]{cx}));
        ancestorsParams.add(Pointer.to(new float[]{cy}));
        params = ancestorsParams.toArray(new NativePointerObject[0]);


    }

    private final int dwell;
    private final float cx;
    private final float cy;

    public int getDwell() {
        return dwell;
    }

    public float getCx() {
        return cx;
    }

    public float getCy() {
        return cy;
    }

    private final NativePointerObject[] params;

    @Override
    public NativePointerObject[] getKernelParams() {
        return params;
    }
}
