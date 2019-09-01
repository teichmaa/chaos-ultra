package cz.cuni.mff.cgg.teichmaa.chaosultra.gui;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.PlaneSegment;
import cz.cuni.mff.cgg.teichmaa.chaosultra.util.FloatPrecision;

import java.util.Collection;
import java.util.List;

public interface GUIModel {
    PlaneSegment getPlaneSegment();

    FloatPrecision getFloatingPointPrecision();

    float getMaxSuperSampling();

    int getMaxIterations();

    int getCanvasWidth();

    int getCanvasHeight();

    Collection<String> getAvailableFractals();

    String getFractalName();

    String getFractalCustomParams();

    List<String> getNewlyLoggedErrors();

    boolean isUseSampleReuse();

    boolean isVisualiseSampleCount();

    boolean isUseAutomaticQuality();

    boolean isUseFoveatedRendering();

    boolean isUseAdaptiveSuperSampling();
}
