package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class Point2DDoubleImmutable {
    double x;
    double y;

    public Point2DDoubleImmutable(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public static Point2DDoubleImmutable of(double x, double y){
        return new Point2DDoubleImmutable(x, y);
    }

    public Point2DDoubleImmutable(Point2DDoubleImmutable original) {
        this.x = original.x;
        this.y = original.y;
    }

    public Point2DDoubleImmutable() {
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }
}
