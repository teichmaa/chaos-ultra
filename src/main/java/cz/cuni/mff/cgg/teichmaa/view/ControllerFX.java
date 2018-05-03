package cz.cuni.mff.cgg.teichmaa.view;

import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLJPanel;
import javafx.embed.swing.SwingNode;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Rectangle2D;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.stage.Screen;

import javax.swing.*;
import java.awt.*;
import java.net.URL;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    @FXML
    SwingNode swingNode;

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

        final GLJPanel gljpanel = new GLJPanel(capabilities);
        RenderingController controller = new RenderingController(width, height, gljpanel);

        gljpanel.addGLEventListener(controller);
        gljpanel.addMouseWheelListener(controller);
        gljpanel.addMouseMotionListener(controller);
        gljpanel.addMouseListener(controller);

        gljpanel.setPreferredSize(new Dimension(width, height));
        final JPanel panel = new JPanel();
        {
            panel.setLayout(new BorderLayout(0,0));
            panel.add(gljpanel);
           // panel.add(new JTextField("rest space"));
        }
        swingNode.setContent(panel);
    }

    public void renderClicked(ActionEvent actionEvent) {

        float x = Float.parseFloat(fractal_x.getText());
        float y = Float.parseFloat(fractal_y.getText());
        float zoom = Float.parseFloat(fractal_zoom.getText());
        int dwell = Integer.parseInt(fractal_dwell.getText());
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
