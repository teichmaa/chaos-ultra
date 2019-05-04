package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

public interface FoveatedRendering {
    //todo screen distance, eyes distance etc

    boolean isUseFoveatedRendering();

    void setUseFoveatedRendering(boolean useFoveatedRendering);

    void setMouseFocus(PointInt mouseFocus);

    PointInt getMouseFocus();

    void setZooming(boolean isZooming);

    boolean isZooming();
}
