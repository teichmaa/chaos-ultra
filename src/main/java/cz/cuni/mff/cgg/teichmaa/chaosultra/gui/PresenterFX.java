package cz.cuni.mff.cgg.teichmaa.chaosultra.gui;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.RenderingController;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.input.MouseEvent;

import javax.swing.*;
import java.io.File;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Date;
import java.util.Objects;
import java.util.ResourceBundle;

public class PresenterFX implements Initializable, GUIPresenter {

    private static final char UNICODE_TIMES_CHAR = '\u00D7';

    @FXML
    private TextArea errorsTextArea;
    @FXML
    private ChoiceBox<String> fractalChoiceBox;
    @FXML
    private TextField fractalCustomParams;
    @FXML
    private Button fractalCustomParamsOKBtn;
    @FXML
    private TitledPane generalParametersPane;
    @FXML
    private Button generalParametersOKBtn;
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
    private CheckBox useFoveatedRendering;
    @FXML
    private CheckBox useSampleReuse;
    @FXML
    private Label precision;

    private RenderingController renderingController;

    private static PresenterFX singleton = null;


    static PresenterFX getSingleton() {
        return singleton;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        if (singleton == null)
            singleton = this;

        fractalChoiceBox.getSelectionModel().selectedItemProperty().addListener((__, oldValue, newValue) -> {
            if (!Objects.equals(oldValue, newValue))
                SwingUtilities.invokeLater(() -> renderingController.onFractalChanged(newValue));
        });


        visualiseSampleCount.selectedProperty().addListener((__, ___, value) -> SwingUtilities.invokeLater(
                () -> renderingController.setVisualiseSampleCount(value))
        );
        useAdaptiveSS.selectedProperty().addListener((__, ___, value) -> SwingUtilities.invokeLater(
                () -> renderingController.setUseAdaptiveSuperSampling(value))
        );
        useAutomaticQuality.selectedProperty().addListener((__, ___, value) -> SwingUtilities.invokeLater(
                () -> renderingController.setAutomaticQuality(value))
        );
        useFoveatedRendering.selectedProperty().addListener((__, ___, value) -> SwingUtilities.invokeLater(
                () -> renderingController.setUseFoveatedRendering(value))
        );
        useSampleReuse.selectedProperty().addListener((__, ___, value) -> SwingUtilities.invokeLater(
                () -> renderingController.setUseSampleReuse(value))
        );

        fractalCustomParamsOKBtn.defaultButtonProperty().bind(fractalCustomParams.focusedProperty());

        generalParametersOKBtn.defaultButtonProperty().bind(
                generalParametersPane.focusedProperty().or(
                        center_x.focusedProperty().or(
                                center_y.focusedProperty().or(
                                        zoom.focusedProperty().or(
                                                maxIterations.focusedProperty().or(
                                                        superSamplingLevel.focusedProperty()
                                                )
                                        )
                                )
                        )
                ));
    }

    void setRenderingController(RenderingController renderingController) {
        Platform.runLater(() -> this.renderingController = renderingController);
    }


    public void showBlockingErrorAlertAsync(String message) {
        Platform.runLater(() -> new Alert(Alert.AlertType.ERROR, message).showAndWait());
    }

    @FXML
    private void renderClicked(ActionEvent actionEvent) {
        render();
    }

    private void render() {
        try {
            double x = Double.parseDouble(center_x.getText());
            double y = Double.parseDouble(center_y.getText());
            double zoom = Double.parseDouble(this.zoom.getText());
            int maxIterations = Integer.parseInt(this.maxIterations.getText());
            int supSamp = Integer.parseInt(superSamplingLevel.getText());
            SwingUtilities.invokeLater(() -> {
                renderingController.setPlaneSegmentRequested(x, y, zoom);
                renderingController.setMaxIterationsRequested(maxIterations);
                renderingController.setSuperSamplingLevelRequested(supSamp);
                renderingController.repaint();
            });
        } catch (NumberFormatException e) {
            showBlockingErrorAlertAsync("Number in a text field could not be parsed.");
        }
    }

    @FXML
    private void defaultViewClicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() ->
                renderingController.showDefaultView()
        );
    }

    @FXML
    private void example1Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.748");
        center_y.setText("0.1");
        zoom.setText("0.0014");
        render();
    }

    @FXML
    private void example2Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.235125");
        center_y.setText("0.827215");
        zoom.setText("4.0E-5");
        render();
    }

    @FXML
    private void example3Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.925");
        center_y.setText("0.266");
        zoom.setText("0.032");
        render();
    }

    @FXML
    private void example4Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.57675236");
        center_y.setText("0.4625193");
        zoom.setText("0.029995363");
        render();
    }

    @FXML
    private void example5Clicked(ActionEvent actionEvent) {
        center_x.setText("-0.551042868375875");
        center_y.setText("0.62714332109057");
        zoom.setText("8.00592947491907E-09");
        maxIterations.setText("3000");
        render();
    }

    @FXML
    private void saveImageClicked(ActionEvent actionEvent) {
        String time = new SimpleDateFormat("dd-MM-YY_HH-mm-ss").format(new Date());
        SwingUtilities.invokeLater(() -> renderingController.saveImageRequested(System.getProperty("user.dir") + File.separator + "saved_images" + File.separator +
                "fractal_" + time + ".png", "png"));
    }

    @FXML
    private void debugButton2Clicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(renderingController::debugRightBottomPixel);
    }

    @FXML
    private void debugButton1Clicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(renderingController::debugFractal);
    }

    @FXML
    private void fractalSpecificParamsSetClicked(ActionEvent actionEvent) {
        SwingUtilities.invokeLater(() -> renderingController.setFractalCustomParams(fractalCustomParams.getText().trim()));
    }

    public static Boolean once = true;

    public void onModelUpdated(GUIModel model) {
        Platform.runLater(() -> {
            if (fractalChoiceBox.getItems().isEmpty()){
                fractalChoiceBox.setItems(FXCollections.observableArrayList(model.getAvailableFractals()).sorted());
            }

            center_x.setText(Double.toString(model.getPlaneSegment().getCenterX()));
            center_y.setText(Double.toString(model.getPlaneSegment().getCenterY()));
            zoom.setText(Double.toString(model.getPlaneSegment().getZoom()));

            superSamplingLevel.setText(Integer.toString(model.getSuperSamplingLevel()));
            maxIterations.setText(Integer.toString(model.getMaxIterations()));

            precision.setText(model.getFloatingPointPrecision().toString());
            dimensions.setText("" + model.getCanvasWidth() + " " + UNICODE_TIMES_CHAR + " " + model.getCanvasHeight());

            fractalCustomParams.setText(model.getFractalCustomParams());

            model.getNewlyLoggedErrors().forEach(c -> errorsTextArea.appendText(timestamp() + c + System.lineSeparator()));
            errorsTextArea.selectPositionCaret(errorsTextArea.getLength()); //scroll to end

            visualiseSampleCount.setSelected(model.isVisualiseSampleCount());
            useSampleReuse.setSelected(model.isUseSampleReuse());
            useAutomaticQuality.setSelected(model.isUseAutomaticQuality());
            useAdaptiveSS.setSelected(model.isUseAdaptiveSuperSampling());
            useFoveatedRendering.setSelected(model.isUseFoveatedRendering());
        });
    }

    private String timestamp() {
        return LocalTime.now().format(timeFormat) + ": ";
    }

    private DateTimeFormatter timeFormat = DateTimeFormatter.ofPattern("HH:mm:ss");

    @FXML
    private void clearErrors(MouseEvent mouseEvent) {
        errorsTextArea.setText("");
    }

    public void reloadFractal(ActionEvent actionEvent) {
        renderingController.reloadFractal();
    }
}
