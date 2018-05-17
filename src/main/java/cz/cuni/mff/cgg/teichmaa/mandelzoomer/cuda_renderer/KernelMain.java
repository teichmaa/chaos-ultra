package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.Pointer;
import jcuda.driver.CUmodule;

class KernelMain extends RenderingKernel {

    /**
     * Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     */
    public static final String name = "fractalRenderMain";

    private final short PARAM_IDX_SUPER_SAMPLING_LEVEL;
    private final short PARAM_IDX_ADAPTIVE_SS;
    private final short PARAM_IDX_VISUALISE_ADAPTIVE_SS;
    final short PARAM_IDX_RANDOM_SAMPLES;
    private final short PARAM_IDX_RENDER_RADIUS;
    private final short PARAM_IDX_FOCUS_X;
    private final short PARAM_IDX_FOCUS_Y;

    KernelMain(CUmodule ownerModule) {
        super(name, ownerModule);

        //initialize params[] :
        PARAM_IDX_SUPER_SAMPLING_LEVEL = addParam(null);
        PARAM_IDX_ADAPTIVE_SS = addParam(null);
        PARAM_IDX_VISUALISE_ADAPTIVE_SS = addParam(null);
        PARAM_IDX_RANDOM_SAMPLES = addParam(null);
        PARAM_IDX_RENDER_RADIUS = addParam(null);
        PARAM_IDX_FOCUS_X = addParam(null);
        PARAM_IDX_FOCUS_Y = addParam(null);

        setSuperSamplingLevel(1);
        setAdaptiveSS(true);
        setVisualiseAdaptiveSS(false);
        setRenderRadius(FractalRenderer.FOVEATION_CENTER_RADIUS);
    }

    private int superSamplingLevel;
    private boolean adaptiveSS;
    private boolean visualiseAdaptiveSS;
    private int renderRadius;
    private int focusx;
    private int focusy;

    boolean isVisualiseAdaptiveSS() {
        return visualiseAdaptiveSS;
    }

    void setVisualiseAdaptiveSS(boolean visualiseAdaptiveSS) {
        this.visualiseAdaptiveSS = visualiseAdaptiveSS;
        params[PARAM_IDX_VISUALISE_ADAPTIVE_SS] = Pointer.to(new int[]{visualiseAdaptiveSS ? 1 : 0});
    }

    boolean isAdaptiveSS() {
        return adaptiveSS;
    }

    void setAdaptiveSS(boolean adaptiveSS) {
        this.adaptiveSS = adaptiveSS;
        params[PARAM_IDX_ADAPTIVE_SS] = Pointer.to(new int[]{adaptiveSS ? 1 : 0});
    }


    void setSuperSamplingLevel(int superSamplingLevel) {
        this.superSamplingLevel = superSamplingLevel;
        params[PARAM_IDX_SUPER_SAMPLING_LEVEL] = Pointer.to(new int[]{superSamplingLevel});
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
        params[PARAM_IDX_RENDER_RADIUS] = Pointer.to(new int[]{renderRadius});
    }

    public void setFocusDefault(){
        setFocus(getWidth() / 2, getHeight() /2);
    }

    public void setFocus(int x, int y){
        params[PARAM_IDX_FOCUS_X] = Pointer.to(new int[]{x});
        params[PARAM_IDX_FOCUS_Y] = Pointer.to(new int[]{y});
        this.focusx = x;
        this.focusy = y;
    }

    public int getFocusx() {
        return focusx;
    }

    public int getFocusy() {
        return focusy;
    }
}
