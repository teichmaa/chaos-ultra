package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;


import com.jogamp.opengl.*;

import com.jogamp.opengl.awt.GLCanvas;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.embed.swing.JFXPanel;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

import javax.swing.*;

import java.awt.*;
import java.io.IOException;

import static jcuda.driver.JCudaDriver.*;

public class Main {

    private static void initFXinSwing(JFXPanel fxPanel) {
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
    }

    private static void createAndShowGUI() {
        //Create and set up the window.
        final JFrame root = new JFrame("Mandelzoomer");
        root.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        final JFXPanel fxPanel = new JFXPanel();
        Platform.runLater(() -> initFXinSwing(fxPanel));

        while (ControllerFX.getSingleton() == null) {
            //TODO TODO TODO this is super bad bad practice
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        root.setLayout(new BoxLayout(root.getContentPane(), BoxLayout.X_AXIS));
        root.add(getGLCanvas(ControllerFX.getSingleton()));
        root.add(fxPanel);

        root.setSize(800, 800);
        root.setVisible(true);
    }

    private static JPanel getGLCanvas(ControllerFX controllerFX) {
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

        final JPanel panel = new JPanel();
        {
            panel.setLayout(new BorderLayout(0, 0));
            panel.add(fractalCanvas);
        }
        return panel;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(Main::createAndShowGUI);
    }

    public static void testCudaEtc() {

        GL2 gl = GLContext.getCurrentGL().getGL2();

        gl.glBegin(GL2.GL_LINES);
        {
            gl.glVertex3f(0f, 0f, 0);
            gl.glVertex3f(-1f, 1f, 0);
        }
        gl.glEnd();
        gl.glFinish();


        int breakpoint = 0;

        JCudaDriver.setExceptionsEnabled(true);

        cuInit(0);

        CUdevice dev = new CUdevice();
        CUcontext ctx = new CUcontext();
        cuDeviceGet(dev, 0);
        cuCtxCreate(ctx, 0, dev);

        int[] deviceCount = new int[1];
        CUdevice[] devices = new CUdevice[10];
        cuGLGetDevices(deviceCount, devices, 10, 1);

    }

}
