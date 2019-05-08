package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

public class PointDoubleImmutable {
    private final double x;
    private final double y;

    public PointDoubleImmutable(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public static PointDoubleImmutable of(double x, double y){
        return new PointDoubleImmutable(x, y);
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }
}
