package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;
import javafx.beans.property.ObjectProperty;

public interface DynamicFloatingPointPrecision {

    public FloatPrecision getFloatingPointPrecision();

    public ObjectProperty<FloatPrecision> floatingPointPrecisionProperty() ;

    public void setFloatingPointPrecision(FloatPrecision floatingPointPrecision);
}
