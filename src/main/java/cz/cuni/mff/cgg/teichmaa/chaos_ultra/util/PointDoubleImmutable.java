package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

public class PointDoubleImmutable {
    private final double x;
    private final double y;

    public PointDoubleImmutable(double x, double y) {
        this.x = x;
        this.y = y;
    }

    @org.jetbrains.annotations.NotNull
    @org.jetbrains.annotations.Contract(value = "_, _ -> new", pure = true)
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
