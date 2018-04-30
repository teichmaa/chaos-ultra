package cz.cuni.mff.cgg.teichmaa.view;

import cz.cuni.mff.cgg.teichmaa.cuda.CudaLauncher;
import cz.cuni.mff.cgg.teichmaa.cuda.MandelbrotKernel;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.WritableImage;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ResourceBundle;

public class Controller implements Initializable {

    private final int FULL_OPACITY_MASK = 0xff000000;

    @FXML
    WritableImage fractalImage;

    @FXML
    Button renderButton;

    @FXML
    private TextField fractal_x;
    @FXML
    private TextField fractal_y;
    @FXML
    private TextField fractal_zoom;
    @FXML
    private TextField fractal_dwell;

    private CudaLauncher fractalRenderer;
    private com.sun.prism.Image prismImg;
    private ByteBuffer theUltimateBuffer;
    private int width;
    private int height;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        width = (int) fractalImage.getWidth();
        height =(int) fractalImage.getHeight();

        initInjectBufferInImage();

        fractalRenderer = new CudaLauncher(new MandelbrotKernel(0, width, height, 0, 0, 0, 0));
    }


    private void initInjectBufferInImage() {

        try {
            int SIZEOF_INT = 4;
            theUltimateBuffer = ByteBuffer.allocateDirect(width * height * SIZEOF_INT);

            // Get the platform image
            Method getWritablePlatformImage = javafx.scene.image.Image.class.getDeclaredMethod("getWritablePlatformImage");
            getWritablePlatformImage.setAccessible(true);
            prismImg = (com.sun.prism.Image) getWritablePlatformImage.invoke(fractalImage);

            // Replace the buffer
            Field pixelBuffer = com.sun.prism.Image.class.getDeclaredField("pixelBuffer");
            pixelBuffer.setAccessible(true);
            pixelBuffer.set(prismImg, theUltimateBuffer);

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    private void invalidateImage() {
        try {
            // Invalidate the platform image
            Field serial = com.sun.prism.Image.class.getDeclaredField("serial");
            serial.setAccessible(true);
            Array.setInt(serial.get(prismImg), 0, Array.getInt(serial.get(prismImg), 0) + 1);

            // Invalidate the WritableImage
            Method pixelsDirty = javafx.scene.image.Image.class.getDeclaredMethod("pixelsDirty");
            pixelsDirty.setAccessible(true);
            pixelsDirty.invoke(fractalImage);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void renderClicked(ActionEvent actionEvent) {

        System.out.println();

        long startTime = System.currentTimeMillis();

        float x = Float.parseFloat(fractal_x.getText());
        float y = Float.parseFloat(fractal_y.getText());
        float zoom = Float.parseFloat(fractal_zoom.getText());
        float windowHeight = 1;
        float windowWidth = windowHeight / (float) height * width;
        float left_bottom_x = x - windowWidth * zoom / 2;
        float left_bottom_y = y - windowHeight * zoom / 2;
        float right_top_x = x + windowWidth * zoom / 2;
        float right_top_y = y + windowHeight * zoom / 2;

        int dwell = Integer.parseInt(fractal_dwell.getText());

        fractalRenderer.getKernel().setDwell(dwell);
        fractalRenderer.getKernel().setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);

        long renderStartTime = System.currentTimeMillis();
        fractalRenderer.launchKernel(theUltimateBuffer, true);
        invalidateImage();

//        fractalImage.getPixelWriter().setPixels(0,0,width, height, PixelFormat.getByteBgraPreInstance(), theUltimateBuffer, width * 4);

        long endTime = System.currentTimeMillis();
        System.out.println("Whole operation done in " + (endTime - startTime) + " ms");
        System.out.println("Rendering done in       " + (endTime - renderStartTime) + " ms");

    }

    public void sample0Clicked(ActionEvent actionEvent) {
        fractal_x.setText("-0.5");
        fractal_y.setText("0");
        fractal_zoom.setText("2");
        renderClicked(actionEvent);
    }

    public void sample1Clicked(ActionEvent actionEvent) {
        fractal_x.setText("-0.748");
        fractal_y.setText("0.1");
        fractal_zoom.setText("0.0014");
        renderClicked(actionEvent);
    }

    public void sample2Clicked(ActionEvent actionEvent) {
        fractal_x.setText("-0.235125");
        fractal_y.setText("0.827215");
        fractal_zoom.setText("4.0E-5");
        renderClicked(actionEvent);
    }

    public void sample3Clicked(ActionEvent actionEvent) {
        fractal_x.setText(" -0.925");
        fractal_y.setText("0.266");
        fractal_zoom.setText("0.032");
        renderClicked(actionEvent);
    }
}
