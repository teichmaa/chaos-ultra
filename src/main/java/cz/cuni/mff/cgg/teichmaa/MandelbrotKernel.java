package cz.cuni.mff.cgg.teichmaa;

import jcuda.NativePointerObject;

import java.util.List;

public class MandelbrotKernel extends AbstractFractalRenderKernel {


    public MandelbrotKernel( int dwell, int width, int height, float left_bottom_x, float left_bottom_y, float right_top_x, float right_top_y) {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\mandelbrot.ptx", "mandelbrot", dwell, width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);
    }

//    @Override
//    public NativePointerObject[] getKernelParams() {
//        //TODO in long term, it is needless to perform this every time I want to render. Try to think of a better pattern that allows the dynamic change of pramaters values.
//        return super.getKernelParamsInternal().toArray(new NativePointerObject[0]);
//    }

}
