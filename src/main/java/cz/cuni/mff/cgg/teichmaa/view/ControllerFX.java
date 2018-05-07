package cz.cuni.mff.cgg.teichmaa.view;

import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLJPanel;
import javafx.embed.swing.SwingNode;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Rectangle2D;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TextField;
import javafx.scene.control.Label;
import javafx.stage.Screen;


import java.awt.Dimension;
import java.awt.BorderLayout;

import javax.sound.midi.Soundbank;
import javax.swing.*;
import java.net.URL;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    @FXML
    SwingNode swingNode;

    @FXML
    Button renderButton;

    private ConcurrentParamHolder params;

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

    private int width;
    private int height;

    @Override
    public void initialize(URL location, ResourceBundle resources) {

        Rectangle2D bounds = Screen.getPrimary().getVisualBounds();

        width = 1000;
        height = 1000;
        GLInit();

    }

    private void GLInit() {
        final GLProfile profile = GLProfile.get(GLProfile.GL2);
        GLCapabilities capabilities = new GLCapabilities(profile);

        fractalCanvas = new GLJPanel(capabilities);
        final ConcurrentParamHolder params = new ConcurrentParamHolder();
        this.params = params;
        RenderingController c = new RenderingController(width, height, fractalCanvas, params, this);
        {
            fractalCanvas.addGLEventListener(c);
            fractalCanvas.addMouseWheelListener(c);
            fractalCanvas.addMouseMotionListener(c);
            fractalCanvas.addMouseListener(c);
        }

        fractalCanvas.setPreferredSize(new Dimension(width, height));
        final JPanel panel = new JPanel();
        {
            panel.setLayout(new BorderLayout(0,0));
            panel.add(fractalCanvas);
           // panel.add(new JTextField("rest space"));
        }
        swingNode.setContent(panel);
    }

    public void renderClicked(ActionEvent actionEvent) {

        float x = Float.parseFloat(fractal_x.getText());
        float y = Float.parseFloat(fractal_y.getText());
        float zoom = Float.parseFloat(fractal_zoom.getText());
        int dwell = Integer.parseInt(fractal_dwell.getText());
        int supsamp = Integer.parseInt(fractal_superSamplingLevel.getText());
        if(supsamp > 256){
            supsamp = 256;
            System.out.println("Warning: super sampling level clamped to " +supsamp+", higher is not supported");
            fractal_superSamplingLevel.setText("" + supsamp);
        }

        params.setXY(x, y);
        params.setDwell(dwell);
        params.setZoom(zoom);
        params.setSuperSamplingLevel(supsamp);
        params.setRequestingRender(true);
        SwingUtilities.invokeLater(fractalCanvas::repaint);
    }

    void updateParams(){
        fractal_x.setText("" + params.getX());
        fractal_y.setText("" + params.getY());
        fractal_dwell.setText("" + params.getDwell());
        fractal_zoom.setText("" + params.getZoom());
        fractal_superSamplingLevel.setText("" + params.getSuperSamplingLevel());
        fractal_adaptiveSS.setSelected(params.isAdaptiveSS());
        dimensions.setText("" + params.getWidth() + " x " + params.getHeight());
    }

    public void loadClicked(ActionEvent actionEvent) {
        updateParams();
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

    public void adaptiveSSselected(ActionEvent actionEvent) {
        boolean val = fractal_adaptiveSS.isSelected();
        params.setAdaptiveSS(val);
        if(!val){
            params.setVisualiseAdaptiveSS(false);
            visualiseAdaptiveSS.setSelected(false);
        }
        params.setRequestingRender(true);
        SwingUtilities.invokeLater(fractalCanvas::repaint);
    }

    public void visualiseAdaptiveSSselected(ActionEvent actionEvent) {
        boolean val = visualiseAdaptiveSS.isSelected();
        params.setVisualiseAdaptiveSS(val);
        if(val){
            params.setAdaptiveSS(true);
            fractal_adaptiveSS.setSelected(true);
        }
        params.setRequestingRender(true);
        SwingUtilities.invokeLater(fractalCanvas::repaint);
    }
}
