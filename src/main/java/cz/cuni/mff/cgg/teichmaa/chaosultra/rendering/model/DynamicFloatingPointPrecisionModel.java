package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

import cz.cuni.mff.cgg.teichmaa.chaosultra.util.FloatPrecision;

public interface DynamicFloatingPointPrecisionModel {

    FloatPrecision getFloatingPointPrecision();

    void setFloatingPointPrecision(FloatPrecision floatingPointPrecision);
}
