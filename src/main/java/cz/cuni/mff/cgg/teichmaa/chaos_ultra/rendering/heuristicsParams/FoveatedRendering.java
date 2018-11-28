package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import javafx.beans.property.BooleanProperty;

public interface FoveatedRendering {
    //todo screen distance, eyes distance etc


    public boolean isUseFoveatedRendering();

    public BooleanProperty useFoveatedRenderingProperty();

    public void setUseFoveatedRendering(boolean useFoveatedRendering);
}
