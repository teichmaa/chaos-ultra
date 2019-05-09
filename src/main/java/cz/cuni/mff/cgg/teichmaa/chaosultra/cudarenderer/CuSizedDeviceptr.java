package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import jcuda.driver.CUdeviceptr;

public class CuSizedDeviceptr {
    private final CUdeviceptr ptr;
    private final long size;

    public CUdeviceptr getPtr() {
        return ptr;
    }

    public long getSize() {
        return size;
    }

    public CuSizedDeviceptr(CUdeviceptr ptr, long size) {
        this.ptr = ptr;
        this.size = size;
    }

    public static CuSizedDeviceptr of(CUdeviceptr ptr, long size){
        return new CuSizedDeviceptr(ptr, size);
    }
}
