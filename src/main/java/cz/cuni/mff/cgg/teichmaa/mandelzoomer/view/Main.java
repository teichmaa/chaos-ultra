package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;


import com.jogamp.opengl.*;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.*;

public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("/window.fxml"));
        primaryStage.setTitle("Mandelzoomer");
        primaryStage.setScene(new Scene(root, 1024, 1024));
        primaryStage.show();
    }


    public static void main(String[] args) {
        launch(args);
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

        return;
    }

}
