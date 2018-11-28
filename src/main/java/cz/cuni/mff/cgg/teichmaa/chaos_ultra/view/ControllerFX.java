package cz.cuni.mff.cgg.teichmaa.chaos_ultra.view;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;


import javax.swing.*;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    static final int SUPER_SAMPLING_MAX_LEVEL = RenderingController.SUPER_SAMPLING_MAX_LEVEL;
    @FXML
    private Button renderButton;
    @FXML
    private TextField center_x;
    @FXML
    private TextField center_y;
    @FXML
    private TextField zoom;
    @FXML
    private TextField maxIterations;
    @FXML
    private TextField superSamplingLevel;
    @FXML
    private Label dimensions;
    @FXML
    private CheckBox useAdaptiveSS;
    @FXML
    private CheckBox visualiseSampleCount;
    @FXML
    private CheckBox useAutomaticQuality;
    @FXML
    private CheckBox useFoveation;
    @FXML
    private CheckBox useSampleReuse;
    @FXML
    private Label precision;

    private RenderingController renderingController;

    private static ControllerFX singleton = null;

    static ControllerFX getSingleton() {
        return singleton;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        if (singleton == null)
            singleton = this;
    }

    void setRenderingController(RenderingController renderingController) {
        Platform.runLater(() -> this.renderingController = renderingController);
    }

    @FXML
    private void renderClicked(ActionEvent actionEvent) {
        useAutomaticQuality.setSelected(false);
        render();
    }

    private void render() {
        try {
            double x = Double.parseDouble(center_x.getText());
            double y = Double.parseDouble(center_y.getText());
            double zoom = Double.parseDouble(this.zoom.getText());
            int maxIterations = Integer.parseInt(this.maxIterations.getText());
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

            SwingUtilities.invokeLater(() -> {
                renderingController.setBounds(x, y, zoom);
                renderingController.setMaxIterations(maxIterations);
                renderingController.setSuperSamplingLevel(supsamp);
                renderingController.repaint();
            });
        } catch (NumberFormatException e) {
            System.out.println("Warning: number in a text field could not be parsed.");
        }
    }

    void setX(double x) {
        Platform.runLater(() -> center_x.setText("" + x));
    }

    void setY(double y) {
        Platform.runLater(() -> center_y.setText("" + y));
    }

    void setSuperSamplingLevel(int SSLevel) {
        Platform.runLater(() -> superSamplingLevel.setText("" + SSLevel));
    }

    void setMaxIterations(int maxIterations) {
        Platform.runLater(() -> this.maxIterations.setText("" + maxIterations));
    }

    void setZoom(double zoom) {
        Platform.runLater(() -> this.zoom.setText("" + zoom));
    }

    void setPrecision(FloatPrecision value) {
        Platform.runLater(() -> precision.setText(value.toString()));
    }

    public void showDefaultView() {
        Platform.runLater(() -> {
            center_x.setText("-0.5");
            center_y.setText("0");
            zoom.setText("2");
            maxIterations.setText("1400");
            superSamplingLevel.setText("5");
            useAdaptiveSS.setSelected(true);
            renderingController.setAdaptiveSS(true);
            useFoveation.setSelected(true);
            renderingController.setUseFoveation(true);
            useSampleReuse.setSelected(true);
            renderingController.setUseSampleReuse(true);
            useAutomaticQuality.setSelected(true);
            renderingController.setUseAutomaticQuality(true);
            visualiseSampleCount.setSelected(false);
            renderingController.setVisualiseSampleCount(false);
            render();
        });
    }

    @FXML
    private void example0Clicked(ActionEvent actionEvent) {
        showDefaultView();
    }

    @FXML
    private void example1Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.748");
        center_y.setText("0.1");
        zoom.setText("0.0014");
        renderClicked(actionEvent);
    }

    @FXML
    private void example2Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.235125");
        center_y.setText("0.827215");
        zoom.setText("4.0E-5");
        renderClicked(actionEvent);
    }

    @FXML
    private void example3Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.925");
        center_y.setText("0.266");
        zoom.setText("0.032");
        renderClicked(actionEvent);
    }

    @FXML
    private void example4Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.57675236");
        center_y.setText("0.4625193");
        zoom.setText("0.029995363");
        superSamplingLevel.setText("32");
        useAdaptiveSS.setSelected(false);
        renderClicked(actionEvent);
    }

    @FXML
    private void example5Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.551042868375875");
        center_y.setText("0.62714332109057");
        zoom.setText("8.00592947491907E-09");
        superSamplingLevel.setText("8");
        useAdaptiveSS.setSelected(false);
        setMaxIterations(3000);
        renderClicked(actionEvent);
    }

    @FXML
    private void adaptiveSSSelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() -> {
            renderingController.setAdaptiveSS(useAdaptiveSS.isSelected());
        });
    }

    @FXML
    private void visualiseSampleCountSelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() -> {
            renderingController.setVisualiseSampleCount(visualiseSampleCount.isSelected());
        });
    }

    @FXML
    private void saveImageClicked(ActionEvent actionEvent) {
        String time = new SimpleDateFormat("dd-MM-YY_HH-mm-ss").format(new Date());
        SwingUtilities.invokeLater(() -> renderingController.saveImage("E:\\Tonda\\Desktop\\fractal-out\\fractal_" + time + ".png", "png"));
    }


    @FXML
    private void automaticQualitySelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() ->
                renderingController.setUseAutomaticQuality(useAutomaticQuality.isSelected())
        );
    }

    public void setDimensions(int width, int height) {
        Platform.runLater(() -> dimensions.setText("" + width + " x " + height));
    }

    @FXML
    private void debugButton2Clicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(renderingController::debugRightBottomPixel);
    }

    @FXML
    private void useFoveationSelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() ->
                renderingController.setUseFoveation(useFoveation.isSelected())
        );
    }

    @FXML
    private void useSampleReuseSelected(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() ->
                renderingController.setUseSampleReuse(useSampleReuse.isSelected())
        );
    }

    @FXML
    private void debugButton1Clicked(ActionEvent actionEvent) {
    }

    void showErrorMessage(String message) {
        Platform.runLater(() -> new Alert(Alert.AlertType.ERROR, message).showAndWait());
    }
}
