package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.*;
import jcuda.runtime.*;

public class FractalRendererMain {

    @Deprecated
    public static void main(String args[]) {

        int dwell = Integer.parseInt(args[0]);
        int width = Integer.parseInt(args[1]);
        int height = Integer.parseInt(args[2]);
        float x = Float.parseFloat(args[3]);
        float y = Float.parseFloat(args[4]);
        float zoom = Float.parseFloat(args[5]);
        boolean saveImage = Boolean.parseBoolean(args[6]);
        String f = args[7].trim();

        float windowHeight = 1;
        float windowWidth = windowHeight / (float) height * width;

        float left_bottom_x = x - windowWidth * zoom / 2;
        float left_bottom_y = y - windowHeight * zoom / 2;
        float right_top_x = x + windowWidth * zoom / 2;
        float right_top_y = y + windowHeight * zoom / 2;


        AbstractFractalRenderKernel kernel;
        if (f.equals("m") || f.equals("mandel") || f.equals("mandelbrot")) {
            kernel = new MandelbrotKernel(dwell, width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);
        } else if (f.equals("j") || f.equals("julia")) {
            kernel = new JuliaKernel(dwell, width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y, Float.parseFloat(args[9]), Float.parseFloat(args[10]));
        } else {
            System.out.println("Unknown function name: " + f);
            return;
        }

        //CudaLauncher r = new CudaLauncher(kernel);
//        r.launchKernel();

    }

    private static void basicTest() {
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: " + pointer);
        JCuda.cudaFree(pointer);

    }
}