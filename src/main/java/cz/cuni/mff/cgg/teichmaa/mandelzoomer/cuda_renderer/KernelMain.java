package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.driver.CUmodule;

abstract class KernelMain extends RenderingKernel {

    private final short PARAM_IDX_SUPER_SAMPLING_LEVEL;
    private final short PARAM_IDX_ADAPTIVE_SS;
    private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS;
    final short PARAM_IDX_RANDOM_SAMPLES;
    private final short PARAM_IDX_RENDER_RADIUS;
    private final short PARAM_IDX_FOCUS_X;
    private final short PARAM_IDX_FOCUS_Y;

    KernelMain(String name, CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize params[] :
        PARAM_IDX_SUPER_SAMPLING_LEVEL = registerParam();
        PARAM_IDX_ADAPTIVE_SS = registerParam();
        PARAM_IDX_VISUALISE_ADAPTIVE_SS = registerParam();
        PARAM_IDX_RANDOM_SAMPLES = registerParam();
        PARAM_IDX_RENDER_RADIUS = registerParam();
        PARAM_IDX_FOCUS_X = registerParam();
        PARAM_IDX_FOCUS_Y = registerParam();

        setSuperSamplingLevel(1);
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
        setRenderRadius(FractalRenderer.FOVEATION_CENTER_RADIUS);
    }

    private int superSamplingLevel;
    private boolean adaptiveSS;
    private boolean visualiseAdaptiveSS;
    private int renderRadius;
    private int focus_x;
    private int focus_y;

    boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = pointerTo(visualiseAdaptiveSS);
    }

    boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = pointerTo(adaptiveSS);
    }


    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = pointerTo(superSamplingLevel);
    }

    int getSuperSamplingLevel() {
        return superSamplingLevel;
    }

    public int getRenderRadius() {
        return renderRadius;
    }

    public void setRenderRadiusToMax(){
        setRenderRadius(Math.max(getWidth(),getHeight()));
    }

    public void setRenderRadius(int renderRadius) {
        this.renderRadius = renderRadius;
        params[PARAM_IDX_RENDER_RADIUS] = pointerTo(renderRadius);
    }

    public void setFocusDefault(){
        setFocus(getWidth() / 2, getHeight() /2);
    }

    public void setFocus(int x, int y){
        params[PARAM_IDX_FOCUS_X] = pointerTo(x);
        params[PARAM_IDX_FOCUS_Y] = pointerTo(y);
        this.focus_x = x;
        this.focus_y = y;
    }

    public int getFocusX() {
        return focus_x;
    }

    public int getFocusY() {
        return focus_y;
    }

}
