package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.NativePointerObject;
import jcuda.driver.CUmodule;

public class KernelBlur extends CudaKernel {

    public static final String name = "blur";

    public KernelBlur(CUmodule ownerModule) {
        super(name, ownerModule);
    }

    @Override
    public NativePointerObject[] getKernelParams() {
        System.err.println("Warning: todo");
        return new NativePointerObject[0];
    }
}
