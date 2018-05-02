package cz.cuni.mff.cgg.teichmaa.view;

import com.jogamp.opengl.*;
import com.jogamp.opengl.awt.GLJPanel;
import javafx.embed.swing.SwingNode;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.WritableImage;

import javax.swing.*;
import java.awt.*;
import java.net.URL;
import java.util.ResourceBundle;

public class ControllerFX implements Initializable {

    private final int FULL_OPACITY_MASK = 0xff000000;

    @FXML
    WritableImage fractalImage;

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

        width = 1024;
        height = 1024;
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
            panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
            panel.add(gljpanel);
           // panel.add(new JTextField("rest space"));
        }
        swingNode.setContent(panel);
    }

    public void renderClicked(ActionEvent actionEvent) {

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

        //todo, tell the info to the kernel:

        //fractalRenderer.getKernel().setDwell(dwell);
        //fractalRenderer.getKernel().setBounds(left_bottom_x, left_bottom_y, right_top_x, right_top_y);

        //fractalRenderer.launchKernel(theUltimateBuffer, true);
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
