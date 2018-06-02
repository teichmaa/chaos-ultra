package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

public class CudaHelpers {
    private static boolean already = false;

    private static CUdevice dev;
    private static CUcontext ctx;

    public static CUdevice getDevice() {
        return dev;
    }
    public static CUcontext getContext() {
        return ctx;
    }

    /**
     * Initialise cuda on current system. May be called multiple times, subsequent calls will be ignored. Thread-safe.
     */
    static synchronized void cudaInit(){
        if(already)
            return;
        already = true;

        // Enable exceptions and omit all subsequent error checks:
        JCudaDriver.setExceptionsEnabled(true);

        JCudaDriver.cuInit(0);

        CUdevice dev = new CUdevice();
        CUcontext ctx = new CUcontext();
        JCudaDriver.cuDeviceGet(dev, 0);
        JCudaDriver.cuCtxCreate(ctx, 0, dev);

        CudaHelpers.dev = dev;
        CudaHelpers.ctx = ctx;
    }

}
