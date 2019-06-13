package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

import cz.cuni.mff.cgg.teichmaa.chaosultra.util.PointInt;

public interface FoveatedRenderingModel {
    //todo screen distance, eyes distance etc

    boolean isUseFoveatedRendering();

    void setUseFoveatedRendering(boolean useFoveatedRendering);

    void setMouseFocus(PointInt mouseFocus);

    PointInt getMouseFocus();

    void setZooming(boolean isZooming);

}
