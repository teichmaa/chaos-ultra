package cz.cuni.mff.cgg.teichmaa.chaos_ultra.view;

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
    private TextField center_x;
    @FXML
    private TextField center_y;
    @FXML
    private TextField zoom;
    @FXML
    private TextField dwell;
    @FXML
    private TextField superSamplingLevel;
    @FXML
    private Label dimensions;
    @FXML
    private CheckBox useAdaptiveSS;
    @FXML
    private CheckBox visualiseAdaptiveSS;
    @FXML
    private CheckBox useAutomaticQuality;

    private RenderingController renderingController;

    private static ControllerFX singleton = null;
    static ControllerFX getSingleton(){
        return singleton;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        if(singleton == null)
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
       useAutomaticQuality.setSelected(false);
       render();
    }

    private void render(){
        try {
            double x = Double.parseDouble(center_x.getText());
            double y = Double.parseDouble(center_y.getText());
            double zoom = Double.parseDouble(this.zoom.getText());
            int dwell = Integer.parseInt(this.dwell.getText());
            int supsamp_tmp = Integer.parseInt(superSamplingLevel.getText());
            if (supsamp_tmp >= SUPER_SAMPLING_MAX_LEVEL) {
                supsamp_tmp = SUPER_SAMPLING_MAX_LEVEL;
                System.out.println("Warning: super sampling level clamped to " + supsamp_tmp + ", higher is not supported");
                superSamplingLevel.setText("" + supsamp_tmp);
            } else if (supsamp_tmp < 1) {
                supsamp_tmp = 1;
                superSamplingLevel.setText("1");
            }
            int supsamp = supsamp_tmp; //for lambda, to pass effectively final value
            boolean autoQuality = useAutomaticQuality.isSelected();
            boolean adaptiveSS = useAdaptiveSS.isSelected();

            SwingUtilities.invokeLater(() -> {
                renderingController.setBounds(x, y, zoom);
                renderingController.setDwell(dwell);
                renderingController.setSuperSamplingLevel(supsamp);
                renderingController.setUseAutomaticQuality(autoQuality);
                renderingController.setAdaptiveSS(adaptiveSS);
                renderingController.repaint();
            });
        }catch (NumberFormatException e){
            System.out.println("Warning: number in a text field could not be parsed.");
        }
    }

    void setX(double x) {
        center_x.setText("" + x);
    }

    void setY(double y) {
        center_y.setText("" + y);
    }

    void setSuperSamplingLevel(int SSLevel){
        superSamplingLevel.setText("" + SSLevel);
    }

    void setDwell(int dwell){
        this.dwell.setText("" + dwell);
    }

    void setZoom(double zoom) {
        this.zoom.setText("" + zoom);
    }

    public void showDefaultView(){
        center_x.setText("-0.5");
        center_y.setText("0");
        zoom.setText("2");
        dwell.setText("1400");
        useAdaptiveSS.setSelected(false);
        superSamplingLevel.setText("5");
        useAutomaticQuality.setSelected(true);
        render();
    }

    public void sample0Clicked(ActionEvent actionEvent) {
        showDefaultView();
    }

    public void sample1Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.748");
        center_y.setText("0.1");
        zoom.setText("0.0014");
        renderClicked(actionEvent);
    }

    public void sample2Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.235125");
        center_y.setText("0.827215");
        zoom.setText("4.0E-5");
        renderClicked(actionEvent);
    }

    public void sample3Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.925");
        center_y.setText("0.266");
        zoom.setText("0.032");
        renderClicked(actionEvent);
    }
    public void sample4Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.57675236");
        center_y.setText("0.4625193");
        zoom.setText("0.029995363");
        superSamplingLevel.setText("32");
        useAdaptiveSS.setSelected(false);
        renderClicked(actionEvent);
    }

    public void adaptiveSSSelected(ActionEvent actionEvent) {
        boolean val = useAdaptiveSS.isSelected();
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

    public void visualiseAdaptiveSSSelected(ActionEvent actionEvent) {
        boolean val = visualiseAdaptiveSS.isSelected();
        SwingUtilities.invokeLater(() -> {
            renderingController.setVisualiseAdaptiveSS(val);
            renderingController.repaint();
            if(val)
                renderingController.setAdaptiveSS(true);
        });
        if (val) {
            useAdaptiveSS.setSelected(true);
        }
    }

    public void saveImageClicked(ActionEvent actionEvent) {
        String time = new SimpleDateFormat("dd-MM-YY_HH-mm-ss").format(new Date());
        SwingUtilities.invokeLater(() -> renderingController.saveImage("E:\\Tonda\\Desktop\\fractal-out\\fractal_" + time + ".png", "png"));
    }


    public void automaticQualitySelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() ->
            renderingController.setUseAutomaticQuality(useAutomaticQuality.isSelected())
        );
    }

    public void setDimensions(int width, int height) {
        dimensions.setText("" + width + " x " + height);
    }
}
