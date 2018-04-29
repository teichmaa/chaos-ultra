package UI;

import cz.cuni.mff.cgg.teichmaa.CudaRenderer;
import cz.cuni.mff.cgg.teichmaa.MandelbrotKernel;
import cz.cuni.mff.cgg.teichmaa.RenderingKernel;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;

import java.net.URL;
import java.util.ResourceBundle;

public class Controller implements Initializable {

    private final int FULL_OPACITY_MASK = 0xff000000;
    private final int HALF_OPACITY_MASK = 0x7f000000;

    @FXML
    WritableImage fractalImage;
    private PixelWriter pw;

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
        pw = fractalImage.getPixelWriter();
        for (int i = 0; i < fractalImage.getWidth(); i++) {
            for (int j = 0; j < fractalImage.getHeight(); j++) {
                pw.setArgb(i,j,FULL_OPACITY_MASK | 255 << 16);
            }
        }
    }

    private void hackLolWhat(){
        Image i;
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

        int[] fractal = CudaRenderer.launch(k, false);

        long kernelFinishedTime = System.currentTimeMillis();

        showBitmap(fractal, width, height);

        long endTime = System.currentTimeMillis();

        System.out.println("The computed fractal showed on screen in " + (endTime - kernelFinishedTime) + " ms");
        System.out.println("Whole operation done in " + (endTime - startTime) + " ms");


    }

    private void showBitmap(int[] data, int width, int height){
        pw.setPixels(0,0, width, height, PixelFormat.getIntArgbInstance(), data, 0, width);
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
