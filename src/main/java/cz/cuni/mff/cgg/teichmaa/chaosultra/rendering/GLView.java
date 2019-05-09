package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering;

import com.jogamp.opengl.GLEventListener;

public interface GLView extends GLEventListener {
    void onFractalChanged(String fractalName);

    void debugRightBottomPixel();

    void launchDebugKernel();

    void saveImageAsync(String fileName, String format);

    void onFractalCustomParamsUpdated();

    void repaint();
}
