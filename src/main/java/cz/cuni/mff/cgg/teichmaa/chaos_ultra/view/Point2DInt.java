package cz.cuni.mff.cgg.teichmaa.chaos_ultra.view;

import java.awt.event.MouseEvent;

public class Point2DInt {
    int x;
    int y;

    public Point2DInt(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Point2DInt() {
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
}
