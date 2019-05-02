package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class PointDouble {
    private double x;
    private double y;

    public static PointDouble of(double x, double y) {
        return new PointDouble(x, y);
    }

    public PointDouble(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public PointDouble copy() {
        return new PointDouble(x, y);
    }

    public void increaseX(double dx){
        x += dx;
    }

    public void increaseY(double dy) {
        y += dy;
    }
}
