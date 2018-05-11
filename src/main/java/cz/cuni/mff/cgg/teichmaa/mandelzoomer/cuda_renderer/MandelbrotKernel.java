package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

public class MandelbrotKernel extends FractalRenderingKernel {

    public MandelbrotKernel() {
        super("E:\\Tonda\\Desktop\\Mandelzoomer\\src\\main\\cuda\\mandelbrot.ptx", "mandelbrot", "init");
    }

}
