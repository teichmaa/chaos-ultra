package cz.cuni.mff.cgg.teichmaa;

import jcuda.NativePointerObject;
import jcuda.Pointer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MandelbrotKernel extends RenderingKernel {

    public final short PARAM_IDX_DWELL;

    public MandelbrotKernel( int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\mandelbrot.ptx", "mandelbrot", width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        this.dwell = dwell;


        List<NativePointerObject> ancestorsParams = super.getKernelParamsInternal();
        PARAM_IDX_DWELL = (short) ancestorsParams.size();
        ancestorsParams.add(Pointer.to(new int[]{dwell}));
        params = ancestorsParams.toArray(new NativePointerObject[0]);
        if(params.length != ancestorsParams.size()){
            throw new AssertException("MandelbrotKernel: params.length != ancestorsParams.size(): " + params.length +", "+ ancestorsParams.size());
        }

    }

    private final int dwell;
    private final NativePointerObject[] params;

    public int getDwell() {
        return dwell;
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        return params;
    }

    @Override
    public String toString() {
        return super.toString() + ", dwell: " + dwell;
    }
}
