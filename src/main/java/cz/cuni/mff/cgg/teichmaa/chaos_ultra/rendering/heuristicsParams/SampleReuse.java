package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import javafx.beans.property.BooleanProperty;

public interface SampleReuse {

    public boolean isUseSampleReuse();

    public BooleanProperty useSampleReuseProperty();

    public void setUseSampleReuse(boolean useSampleReuse);
}
