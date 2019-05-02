package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointInt;

public interface FoveatedRendering {
    //todo screen distance, eyes distance etc

    boolean isUseFoveatedRendering();

    void setUseFoveatedRendering(boolean useFoveatedRendering);

    void setFocus(PointInt focus);

    PointInt getFocus();

    void setZooming(boolean isZooming);

    boolean isZooming();
}
