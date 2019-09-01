package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

public class PointDoubleImmutable implements PointDoubleReadable {
    private final double x;
    private final double y;

    public PointDoubleImmutable(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public static PointDoubleImmutable of(double x, double y){
        return new PointDoubleImmutable(x, y);
    }

    @Override
    public double getX() {
        return x;
    }

    @Override
    public double getY() {
        return y;
    }
}
