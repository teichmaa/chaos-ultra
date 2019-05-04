package cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model.PlaneSegment;
import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;

import java.util.Collection;

public interface GUIModel {
    PlaneSegment getPlaneSegment();

    FloatPrecision getFloatingPointPrecision();

    int getSuperSamplingLevel();

    int getMaxIterations();

    int getCanvasWidth();

    int getCanvasHeight();

    Collection<String> getAvailableFractals();

    String getFractalName();
}
