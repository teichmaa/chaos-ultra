package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.util.PointDouble;

/**
 * Represents a rectangular segment of the 2D real plane
 */
public class PlaneSegment {
    private final PointDouble leftBottom;
    private final PointDouble rightTop;

    public PlaneSegment(PointDouble leftBottom, PointDouble rightTop) {
        this.leftBottom = leftBottom;
        this.rightTop = rightTop;
    }

    public PlaneSegment() {
        leftBottom = new PointDouble(0,0);
        rightTop = new PointDouble(0,0);
    }

    public PointDouble getLeftBottom() {
        return leftBottom;
    }

    public PointDouble getRightTop() {
        return rightTop;
    }

    public void setAll(double leftBottomX, double leftBottomY, double rightTopX, double rightTopY) {
        getLeftBottom().setX(leftBottomX);
        getLeftBottom().setY(leftBottomY);
        getRightTop().setX(rightTopX);
        getRightTop().setY(rightTopY);
    }


    public PlaneSegment copy() {
        return new PlaneSegment(this.leftBottom.copy(), this.rightTop.copy());
    }

    /**
     * increase the value of leftBottom.X and rightTop.X
     * @param dx value to add
     */
    public void increaseXsBy(double dx){
        getLeftBottom().increaseX(dx);
        getRightTop().increaseX(dx);
    }

    /**
     * increase the value of leftBottom.Y and rightTop.Y
     * @param dy value to add
     */
    public void increaseYsBy(double dy) {
        getLeftBottom().increaseY(dy);
        getRightTop().increaseY(dy);
    }

    public double getSegmentWidth() {
        return getRightTop().getX() - getLeftBottom().getX();
    }

    public double getSegmentHeight() {
        return getRightTop().getY() - getLeftBottom().getY();
    }

    public double getCenterX() {
        return getLeftBottom().getX() + getSegmentWidth() / 2;
    }

    public double getCenterY() {
        return getLeftBottom().getY() + getSegmentHeight() / 2;
    }

    public double getZoom(){
        return getSegmentHeight();
    }

}
