package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

public class ModuleMandelbrot extends FractalRenderingModule {

    private static String absolutePrefix = "E:\\Tonda\\Desktop\\Mandelzoomer\\";
    public ModuleMandelbrot() {

        super(absolutePrefix+"src\\main\\cuda\\mandelbrot.ptx", "mandelbrot");
    }

}
