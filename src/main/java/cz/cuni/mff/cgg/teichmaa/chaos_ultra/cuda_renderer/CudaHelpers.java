package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.Pointer;
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

    static Pointer pointerTo(int value){
        return Pointer.to(new int[]{value});
    }
    static Pointer pointerTo(long value){
        return Pointer.to(new long[]{value});
    }
    static Pointer pointerTo(float value){
        return Pointer.to(new float[]{value});
    }
    static Pointer pointerTo(double value){
        return Pointer.to(new double[]{value});
    }
    static Pointer pointerTo(boolean value){
        return Pointer.to(new int[]{value ? 1 : 0});
    }
    static Pointer pointerTo(float v0, float v1){
        return Pointer.to(new float[]{v0, v1});
    }
    static Pointer pointerTo(float v0, float v1,float v2, float v3){
        return Pointer.to(new float[]{v0, v1, v2, v3});
    }
    static Pointer pointerTo(double v0, double v1){
        return Pointer.to(new double[]{v0, v1});
    }
    static Pointer pointerTo(double v0, double v1,double v2, double v3){
        return Pointer.to(new double[]{v0, v1, v2, v3});
    }
    static Pointer pointerTo(int v0, int v1){
        return Pointer.to(new int[]{v0, v1});
    }
    static Pointer pointerTo(int v0, int v1,int v2, int v3){
        return Pointer.to(new int[]{v0, v1, v2, v3});
    }
    static Pointer pointerTo(long v0, long v1){
        return Pointer.to(new long[]{v0, v1});
    }
    static Pointer pointerTo(long v0, long v1,long v2, long v3){
        return Pointer.to(new long[]{v0, v1, v2, v3});
    }

}
