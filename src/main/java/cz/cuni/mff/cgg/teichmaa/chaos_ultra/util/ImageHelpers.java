package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class ImageHelpers {

    public static int[] createColorPalette() {
        int max = 256;
        int fullColor = 255;
        int[] p = new int[max * 6];

        //rgb-2-hsv transitions palette:

        //  resp. allow user modulo-rotating of color palette or some other changes

        for (int i = max * 0; i < max * 1; i++) {
            p[i] = fromRGB(i,0,Math.min(fullColor, fullColor/2+i)); //blue to magenta gradient
        }

        for (int i = max * 1; i < max * 2; i++) {
            p[i] = fromRGB(fullColor, 0, fullColor-i); //magenta to red gradient
        }

        for (int i = max * 2; i < max * 3; i++) {
            p[i] = fromRGB(fullColor, i, 0); //red to yellow gradient
        }

        for (int i = max * 3; i < max * 4; i++) {
            p[i] = fromRGB(fullColor-i,fullColor,0); //yellow to green gradient
        }

        for (int i = max * 4; i < max * 5; i++) {
            p[i] = fromRGB(0,fullColor,i); //green to cyan gradient
        }

        for (int i = max * 5; i < max * 6; i++) {
            p[i] = fromRGB(0,fullColor-i,fullColor); //cyan to blue gradient
        }

        return p;
    }

    private static int fromRGB(int r, int g, int b) {
        int a = 255;
        //to RGBA
        int r_shift = 0;
        int g_shift = 8;
        int b_shift = 16;
        int a_shift = 24;
        r <<= r_shift;
        g <<= g_shift;
        b <<= b_shift;
        a <<= a_shift;
        int rMask = 0x000000ff << r_shift;
        int gMask = 0x000000ff << g_shift;
        int bMask = 0x000000ff << b_shift;
        int aMask = 0x000000ff << a_shift;
        r &= rMask;
        g &= gMask;
        b &= bMask;
        a &= aMask;
        return r | g | b | a;
    }


    public static int[][] arrayFlatTo2D(int[] a, int width, int height) {
        int result2D[][] = new int[width][height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                result2D[i][j] = a[j * width + i];
            }
        }
        return result2D;
    }

    public static int[] arrayFlatten(int[][] a) {
        int width = a.length;
        int height = a[0].length;
        int[] result = new int[width * height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                result[j * width + i] = a[i][j];
            }
        }
        return result;
    }

    public static void createImage(int[] rgbs, int width, int height, String fileName, String formatName) {


        DataBuffer rgbData = new DataBufferInt(rgbs, rgbs.length);

        WritableRaster raster = Raster.createPackedRaster(rgbData, width, height, width,
                new int[]{0xff0000, 0xff00, 0xff},
                null);

        ColorModel colorModel = new DirectColorModel(24, 0xff0000, 0xff00, 0xff);

        BufferedImage img = new BufferedImage(colorModel, raster, false, null);

        //System.out.println("writing image to " + fileName);
        try {
            File f = new File(fileName);
            Files.createDirectories(f.getParentFile().toPath());
            ImageIO.write(img, formatName, f);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static void printArray(int[][] a) {
        for (int i = 0; i < a.length; i++) {
            System.out.print(" | ");
            for (int j = 0; j < a[i].length; j++) {
                System.out.print(a[i][j] + " | ");
            }
            System.out.println();
            System.out.println("________________");
            System.out.println();
        }
    }
}
