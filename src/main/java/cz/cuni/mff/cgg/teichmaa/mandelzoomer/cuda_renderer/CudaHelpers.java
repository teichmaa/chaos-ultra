package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;

public class CudaHelpers {
    private static boolean already = false;

    private static CUdevice dev;

    public CUdevice getDevice() {
        return dev;
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

        cuInit(0);

        CUdevice dev = new CUdevice();
        CUcontext ctx = new CUcontext();
        cuDeviceGet(dev, 0);
        cuCtxCreate(ctx, 0, dev);

        CudaHelpers.dev = dev;
    }
}
