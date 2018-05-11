package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

public class MandelbrotKernel extends FractalRenderingKernel {

    private static String absolutePrefix = "E:\\Tonda\\Desktop\\Mandelzoomer\\";
    public MandelbrotKernel() {

        super(absolutePrefix+"src\\main\\cuda\\mandelbrot.ptx", "mandelbrot", "init");
    }

}
