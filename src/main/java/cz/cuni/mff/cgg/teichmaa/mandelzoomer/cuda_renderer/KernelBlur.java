package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.driver.CUmodule;

public class KernelBlur extends CudaKernel {

    public static final String name = "blur";

    public KernelBlur(CUmodule ownerModule) {
        super(name, ownerModule);
    }

}
