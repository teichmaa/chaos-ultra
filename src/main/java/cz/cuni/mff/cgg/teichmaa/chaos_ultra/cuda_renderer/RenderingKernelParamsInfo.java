package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

public class RenderingKernelParamsInfo {
    double left_bottom_x;
    double left_bottom_y;
    double right_top_x;
    double right_top_y;

    void setFrom(RenderingKernel k){
        left_bottom_x = k.getLeft_bottom_x();
        left_bottom_y = k.getLeft_bottom_y();
        right_top_x = k.getRight_top_x();
        right_top_y = k.getRight_top_y();
    }
}
