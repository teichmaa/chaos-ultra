package cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui;

import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

class SceneBuilder {

    public void start() {
        Platform.runLater(this::initialiseFxPanel);
    }

    private final JFXPanel fxPanel = new JFXPanel();

    private void initialiseFxPanel() {
        Parent root;
        try {
            root = FXMLLoader.load(Main.class.getResource("/window.fxml"));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
            return;
        }
        Scene scene = new Scene(root);
        fxPanel.setScene(scene);

        //assuming that the FXMLLoader properly instantiated the ControllerFX singleton
        assert ControllerFX.getSingleton() != null;

        SwingUtilities.invokeLater(this::composeAndRunSwingScene);
    }

    private void composeAndRunSwingScene(){
        final JFrame root = new JFrame("Mandelzoomer");
        root.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        //todo mozna muzu pouzit jiny layout a to vyresi moje problemy
        root.setLayout(new BorderLayout());
        root.add(getGLCanvas(ControllerFX.getSingleton()));
        root.add(fxPanel, BorderLayout.EAST);
        root.setMinimumSize(new Dimension(800, 800)); //todo tohle je blby, ja bych ten JPanel potreboval ctvercovy
        //napriklad https://stackoverflow.com/questions/16075022/making-a-jpanel-square
        // ale to chci resit az na desktopu
        root.setVisible(true);
    }

    private JPanel getGLCanvas(ControllerFX controllerFX) {
        final GLProfile profile = GLProfile.get(GLProfile.GL2);
        final GLCapabilities capabilities = new GLCapabilities(profile);

        final GLCanvas fractalCanvas = new GLCanvas(capabilities);
        final RenderingController renderingController = new RenderingController(fractalCanvas, controllerFX);
        {
            fractalCanvas.addGLEventListener(renderingController);
            fractalCanvas.addMouseWheelListener(renderingController);
            fractalCanvas.addMouseMotionListener(renderingController);
            fractalCanvas.addMouseListener(renderingController);
        }
        controllerFX.setRenderingController(renderingController);

        final JPanel panel = new JPanel();
        {
            panel.setLayout(new BorderLayout(0, 0));
            panel.add(fractalCanvas);
        }
        return panel;
    }

}


