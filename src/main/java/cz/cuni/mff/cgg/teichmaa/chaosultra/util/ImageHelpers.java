package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

import javax.imageio.ImageIO;
import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ImageHelpers {

    public static void main(String[] args) {
        createCustomPalette();
    }

    /**
     * Simple util for creation of gradient palettes
     */
    public static void createCustomPalette() {
        int w = 256 * 2;
        int[] result = new int[w];

        for (int i = 0; i < w / 2; i++) {
            int r = i;
            int g = 0;
            int b = 0;
            result[i] = fromRGBtoRGBA(r, g, b);
            result[w - i - 1] = fromRGBtoRGBA(r, g, b);
        }

        String fileName = "defaultPalette" + ".png";
        saveImageToFile(result, result.length, 1, fileName, "png");

        System.out.println("Saved " + fileName);
    }

    /**
     * Tries to load palette from filePath. If not possible, creates a default palette.
     *
     * @param filePath path of the file to load the palette from
     * @return Array of integers containing either the default palette or the first row of the image. In RGBA format (Red is the least significant).
     */
    public static int[] loadColorPaletteOrDefault(String filePath) {
        if (null == filePath || filePath.isEmpty())
            return createDefaultColorPalette();
        if (!Files.exists(Paths.get(filePath)))
            return createDefaultColorPalette();
        int[] resultOrNull = loadColorPaletteFromFile(filePath);
        if (resultOrNull == null)
            return createDefaultColorPalette();
        else return resultOrNull;
    }

    /**
     * @param filePath path of the file to load the palette from
     * @return null (if IOError) or array of integers containing the first row of the image. In RGBA format (Red is the least significant).
     */
    public static int[] loadColorPaletteFromFile(String filePath) {
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(filePath));
        } catch (IOException e) {
            System.err.println("Error while loading palette " + filePath);
            e.printStackTrace();
            return null;
        }

        int w = image.getWidth();
        int[] result = new int[image.getWidth()];
        image.getRGB(0, 0, w, 1, result, 0, w);
        return fromBGRAtoRGBA(result);
    }

    /**
     * @return Color palette with colorful linear gradients in RGBA format (Red is the least significant).
     */
    public static int[] createDefaultColorPalette() {
        int max = 256;
        int fullColor = 255;
        int[] p = new int[max * 6];

        //rgb-2-hsv transitions palette:

        //  resp. allow user modulo-rotating of color palette or some other changes

        for (int i = max * 0; i < max * 1; i++) {
            p[i] = fromRGBtoRGBA(i, 0, Math.min(fullColor, fullColor / 2 + i)); //blue to magenta gradient
        }

        for (int i = max * 1; i < max * 2; i++) {
            p[i] = fromRGBtoRGBA(fullColor, 0, fullColor - i); //magenta to red gradient
        }

        for (int i = max * 2; i < max * 3; i++) {
            p[i] = fromRGBtoRGBA(fullColor, i, 0); //red to yellow gradient
        }

        for (int i = max * 3; i < max * 4; i++) {
            p[i] = fromRGBtoRGBA(fullColor - i, fullColor, 0); //yellow to green gradient
        }

        for (int i = max * 4; i < max * 5; i++) {
            p[i] = fromRGBtoRGBA(0, fullColor, i); //green to cyan gradient
        }

        for (int i = max * 5; i < max * 6; i++) {
            p[i] = fromRGBtoRGBA(0, fullColor - i, fullColor); //cyan to blue gradient
        }

        return p;
    }

    /**
     *
     * @param input colors in BGRA (little endian), i.e. when read by human: aa rr gg bb
     * @return colors in RGBA (little endian), i.e. when read by human: aa bb gg rr
     */
    private static int[] fromBGRAtoRGBA(int[] input) {
        int[] output = new int[input.length];
        for (int i = 0; i < input.length; i++) {
            int original = input[i];
            int updated;
            int alpha = (original >> 24);
            updated = Integer.reverseBytes(original);
            updated >>= 8;
            updated |= ((alpha << 24) & 0xff000000);
            output[i] = updated;
        }
        return output;
    }

    /**
     * @param r red
     * @param g green
     * @param b blue
     * @return colors in RGBA (little endian), i.e. when read by human: aa bb gg rr
     */
    private static int fromRGBtoRGBA(int r, int g, int b) {
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
        int[][] result2D = new int[width][height];
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

    /**
     * Writes rgb data to a file, as image. If the file (or the path to it) does not exist, it is created.
     * @param rgbs data in RGBA (little endian) (i.e. Red is the least significant), stored row by row
     * @param width image width
     * @param height image height
     * @param fileName file name, including extension
     * @param formatName image format name, e.g. png or jpg
     */
    public static void saveImageToFile(int[] rgbs, int width, int height, String fileName, String formatName) {


        DataBuffer rgbData = new DataBufferInt(rgbs, rgbs.length);

        WritableRaster raster = Raster.createPackedRaster(rgbData, width, height, width,
                new int[]{0xff, 0xff00, 0xff0000},
                null);

        ColorModel colorModel = new DirectColorModel(24, 0xff, 0xff00, 0xff0000);

        BufferedImage img = new BufferedImage(colorModel, raster, false, null);

        try {
            File f = new File(fileName);
            if (f.getParentFile() != null)
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
