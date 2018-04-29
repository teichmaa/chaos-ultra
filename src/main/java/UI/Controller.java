package UI;

import cz.cuni.mff.cgg.teichmaa.CudaRenderer;
import cz.cuni.mff.cgg.teichmaa.MandelbrotKernel;
import cz.cuni.mff.cgg.teichmaa.RenderingKernel;
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

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        initInjectBufferInImage();
        CudaRenderer.init(1280,1024);
    }


    private com.sun.prism.Image prismImg;
    private ByteBuffer theUltimateBuffer;

    private void initInjectBufferInImage() {

        try {
            theUltimateBuffer = ByteBuffer.allocateDirect(1280 * 1024 * 4);

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
        int dwell = Integer.parseInt(fractal_dwell.getText());

        int width = 1280;
        int height = 1024;

        float windowHeight = 1;
        float windowWidth = windowHeight / (float) height * width;

        float left_bottom_x = x - windowWidth * zoom / 2;
        float left_bottom_y = y - windowHeight * zoom / 2;
        float right_top_x = x + windowWidth * zoom / 2;
        float right_top_y = y + windowHeight * zoom / 2;

        RenderingKernel k = new MandelbrotKernel(dwell, width, height, left_bottom_x, left_bottom_y, right_top_x, right_top_y);

        CudaRenderer.launch(k, false, theUltimateBuffer);

        long kernelFinishedTime = System.currentTimeMillis();

       // showBitmap(fractal, width, height);
        invalidateImage();

        long endTime = System.currentTimeMillis();

        System.out.println("The computed fractal showed on screen in " + (endTime - kernelFinishedTime) + " ms");
        System.out.println("Whole operation done in " + (endTime - startTime) + " ms");


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
