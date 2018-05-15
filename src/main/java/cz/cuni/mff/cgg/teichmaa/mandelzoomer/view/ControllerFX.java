package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;

import com.jogamp.opengl.awt.GLJPanel;
import javafx.embed.swing.SwingNode;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.control.Label;


import javax.swing.*;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    static final int SUPER_SAMPLING_MAX_LEVEL = RenderingController.SUPER_SAMPLING_MAX_LEVEL;
    @FXML
    private SwingNode swingNode;
    @FXML
    private Button renderButton;
    private GLJPanel fractalCanvas;
    @FXML
    private TextField fractal_x;
    @FXML
    private TextField fractal_y;
    @FXML
    private TextField fractal_zoom;
    @FXML
    private TextField fractal_dwell;
    @FXML
    private TextField fractal_superSamplingLevel;
    @FXML
    private Label dimensions;
    @FXML
    private CheckBox fractal_adaptiveSS;
    @FXML
    private CheckBox visualiseAdaptiveSS;

    private RenderingController renderingController;

    private static ControllerFX singleton = null;
    static ControllerFX getSingleton(){
        return singleton;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        singleton = this;
        while(RenderingController.getSingleton() == null){
            //TODO TODO TODO this is super bad bad practice
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        renderingController = RenderingController.getSingleton();
    }

    public void renderClicked(ActionEvent actionEvent) {

        float x = Float.parseFloat(fractal_x.getText());
        float y = Float.parseFloat(fractal_y.getText());
        float zoom = Float.parseFloat(fractal_zoom.getText());
        int dwell = Integer.parseInt(fractal_dwell.getText());
        int supsamp_tmp = Integer.parseInt(fractal_superSamplingLevel.getText());
        if (supsamp_tmp >= SUPER_SAMPLING_MAX_LEVEL) {
            supsamp_tmp = SUPER_SAMPLING_MAX_LEVEL;
            System.out.println("Warning: super sampling level clamped to " + supsamp_tmp + ", higher is not supported");
            fractal_superSamplingLevel.setText("" + supsamp_tmp);
        }
        int supsamp = supsamp_tmp; //for lambda, to pass effectively final value

        SwingUtilities.invokeLater(() -> {
            renderingController.setBounds(x, y, zoom);
            renderingController.setDwell(dwell);
            renderingController.setSuperSamplingLevel(supsamp);
            renderingController.repaint();
        });
    }

    void setX(float x) {
        fractal_x.setText("" + x);
    }

    void setY(float y) {
        fractal_y.setText("" + y);
    }

    void setSuperSamplingLevel(int SSlevel){
        fractal_superSamplingLevel.setText("" + SSlevel);
    }

    void setDwell(int dwell){
        fractal_dwell.setText("" + dwell);
    }

    void setZoom(float zoom) {
        fractal_zoom.setText("" + zoom);
    }

    public void showDefaultView(){
        fractal_x.setText("-0.5");
        fractal_y.setText("0");
        fractal_zoom.setText("2");
        fractal_dwell.setText("1400");
        fractal_superSamplingLevel.setText("5");
        renderClicked(null);
    }

    public void sample0Clicked(ActionEvent actionEvent) {
        showDefaultView();
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
        fractal_x.setText("-0.925");
        fractal_y.setText("0.266");
        fractal_zoom.setText("0.032");
        renderClicked(actionEvent);
    }
    public void sample4Clicked(ActionEvent actionEvent) {
        fractal_x.setText("-0.57675236");
        fractal_y.setText("0.4625193");
        fractal_zoom.setText("0.029995363");
        fractal_superSamplingLevel.setText("32");
        fractal_adaptiveSS.setSelected(false);
        renderClicked(actionEvent);
    }

    public void adaptiveSSselected(ActionEvent actionEvent) {
        boolean val = fractal_adaptiveSS.isSelected();
        SwingUtilities.invokeLater(() -> {
            renderingController.setAdaptiveSS(val);
            renderingController.repaint();
            if (!val)
                renderingController.setVisualiseAdaptiveSS(false);
        });
        if (!val) {
            visualiseAdaptiveSS.setSelected(false);
        }
    }

    public void visualiseAdaptiveSSselected(ActionEvent actionEvent) {
        boolean val = visualiseAdaptiveSS.isSelected();
        SwingUtilities.invokeLater(() -> {
            renderingController.setVisualiseAdaptiveSS(val);
            renderingController.repaint();
            if(val)
                renderingController.setAdaptiveSS(true);
        });
        if (val) {
            fractal_adaptiveSS.setSelected(true);
        }
    }

    public void saveImageClicked(ActionEvent actionEvent) {
        String time = new SimpleDateFormat("dd-MM-YY_HH-mm-ss").format(new Date());
        SwingUtilities.invokeLater(() -> renderingController.saveImage("E:\\Tonda\\Desktop\\fractal-out\\fractal_" + time + ".png", "png"));
    }


}
