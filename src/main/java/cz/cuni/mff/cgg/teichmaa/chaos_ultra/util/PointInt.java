package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

import java.awt.event.MouseEvent;

public class PointInt {
    private int x;
    private int y;

    public PointInt(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public PointInt() {
    }

    public int getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public void setXYFrom(MouseEvent e){
        setX(e.getX());
        setY(e.getY());
    }

    public PointInt copy() {
        return new PointInt(x, y);
    }
}
