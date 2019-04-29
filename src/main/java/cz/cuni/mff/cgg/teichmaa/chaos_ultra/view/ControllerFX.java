package cz.cuni.mff.cgg.teichmaa.chaos_ultra.view;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams.ChaosUltraRenderingParams;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.util.converter.NumberStringConverter;


import javax.swing.*;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    static final int SUPER_SAMPLING_MAX_LEVEL = RenderingController.SUPER_SAMPLING_MAX_LEVEL;
    @FXML
    private TextField fractalSpecificParams;
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


    void showErrorMessage(String message) {
        Platform.runLater(() -> new Alert(Alert.AlertType.ERROR, message).showAndWait());
    }

    void bindParamsTo(ChaosUltraRenderingParams params){
        Bindings.bindBidirectional(maxIterations.textProperty(), params.maxIterationsProperty(), new NumberStringConverter());
        Bindings.bindBidirectional(superSamplingLevel.textProperty(), params.superSamplingLevelProperty(), new NumberStringConverter());
        params.useAdaptiveSupersamplingProperty().bindBidirectional(useAdaptiveSS.selectedProperty());
        params.useFoveatedRenderingProperty().bindBidirectional(useFoveation.selectedProperty());
        params.useSampleReuseProperty().bindBidirectional(useSampleReuse.selectedProperty());
        params.visualiseSampleCountProperty().bindBidirectional(visualiseSampleCount.selectedProperty());
        params.automaticQualityProperty().bindBidirectional(useAutomaticQuality.selectedProperty());
    }

    void setX(double x) {
        Platform.runLater(() -> center_x.setText("" + x));
    }

    void setY(double y) {
        Platform.runLater(() -> center_y.setText("" + y));
    }

    void setZoom(double zoom) {
        Platform.runLater(() -> this.zoom.setText("" + zoom));
    }

    public void setDimensions(int width, int height) {
        Platform.runLater(() -> dimensions.setText("" + width + " x " + height));
    }

    void showDefaultView() {
        Platform.runLater(() -> {
            center_x.setText("-0.5");
            center_y.setText("0");
            zoom.setText("2");
            maxIterations.setText("1400");
            superSamplingLevel.setText("5");
            useAdaptiveSS.setSelected(true);
            useFoveation.setSelected(true);
            useSampleReuse.setSelected(true);
            useAutomaticQuality.setSelected(true);
            visualiseSampleCount.setSelected(false);
            render();
        });
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
        maxIterations.setText("3000");
        renderClicked(actionEvent);
    }

    @FXML
    private void saveImageClicked(ActionEvent actionEvent) {
        String time = new SimpleDateFormat("dd-MM-YY_HH-mm-ss").format(new Date());
        SwingUtilities.invokeLater(() -> renderingController.saveImage("E:\\Tonda\\Desktop\\rendering-out\\fractal_" + time + ".png", "png"));
        // todo vybiratko na soubory
    }


    @FXML
    private void debugButton2Clicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(renderingController::debugRightBottomPixel);
    }

    @FXML
    private void debugButton1Clicked(ActionEvent actionEvent) {
    }

    @FXML
    private void fractalSpecificParamsSetClicked(ActionEvent actionEvent) {
        renderingController.setFractalSpecificParams(fractalSpecificParams.getText().trim());
    }
}
