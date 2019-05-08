package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.FloatPrecision;

public interface DynamicFloatingPointPrecisionModel {

    FloatPrecision getFloatingPointPrecision();

    void setFloatingPointPrecision(FloatPrecision floatingPointPrecision);
}
