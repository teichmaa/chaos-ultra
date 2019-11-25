package cz.cuni.mff.cgg.teichmaa.chaosultra.gui;
import javax.swing.plaf.ColorUIResource;
import javax.swing.plaf.metal.DefaultMetalTheme;
import javax.swing.plaf.metal.MetalLookAndFeel;

import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLProfile;
import com.jogamp.opengl.awt.GLCanvas;
import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.RenderingController;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

class SceneBuilder {

    public static final String APP_TITLE = "chaos-ultra";

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

        //assuming that the FXMLLoader properly instantiated the PresenterFX singleton
        assert PresenterFX.getSingleton() != null;

        SwingUtilities.invokeLater(this::composeAndRunSwingScene);
    }

    private void composeAndRunSwingScene(){
        final JFrame root = new JFrame(APP_TITLE);
        root.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        root.setLayout(new BorderLayout());
        root.add(getGLCanvas(PresenterFX.getSingleton()));
        root.add(fxPanel, BorderLayout.EAST);
        root.setMinimumSize(new Dimension(500, 350));
        root.setSize(1280 + 291, 720 + 39);
        root.setVisible(true);
    }

    private JPanel getGLCanvas(PresenterFX presenterFX) {
        final GLProfile profile = GLProfile.get(GLProfile.GL2);
        final GLCapabilities capabilities = new GLCapabilities(profile);

        final GLCanvas fractalCanvas = new GLCanvas(capabilities);
        final RenderingController renderingController = new RenderingController(fractalCanvas, presenterFX);
        {
            fractalCanvas.addGLEventListener(renderingController.getView());
            fractalCanvas.addMouseWheelListener(renderingController);
            fractalCanvas.addMouseMotionListener(renderingController);
            fractalCanvas.addMouseListener(renderingController);
        }
        presenterFX.setRenderingController(renderingController);

        final JPanel panel = new JPanel();
        {
            panel.setLayout(new BorderLayout(0, 0));
            panel.add(fractalCanvas);
        }
        return panel;
    }

}


