package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.heuristicsParams;

import javafx.beans.property.IntegerProperty;

public interface IterationLimit {


    public int getMaxIterations();

    public IntegerProperty maxIterationsProperty();

    public void setMaxIterations(int maxIterations);
}
