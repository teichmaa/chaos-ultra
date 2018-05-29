package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

public class ModuleMandelbrot extends FractalRenderingModule {

    private static String absolutePrefix = "E:\\Tonda\\Desktop\\Mandelzoomer\\";
    public ModuleMandelbrot() {

        super(absolutePrefix+"src\\main\\cuda\\mandelbrot.ptx", "mandelbrot");
    }

}
