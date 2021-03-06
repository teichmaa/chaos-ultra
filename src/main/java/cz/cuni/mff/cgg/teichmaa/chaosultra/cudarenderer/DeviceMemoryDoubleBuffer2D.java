package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import jcuda.CudaException;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

import java.io.Closeable;

//TODO a better design would be to have a separate class for Cuda2DArray, containing its CUdeviceptr, width, height and pitch. Then, here rename all the methods (no need for the '2D' indication). This is like programming C, instead  of C++ (whilst having Java)


/**
 * Represents two distinct 2D arrays in CUDA memory, whose order may be switched
 */
class DeviceMemoryDoubleBuffer2D implements Closeable {

    private CUdeviceptr array1;
    private CUdeviceptr array2;
    private long array1Pitch;
    private long array2Pitch;
    private final int elementSize;

    /**
     * @param elementSize how many 4-byte elements to allocate per one cell
     */
    public DeviceMemoryDoubleBuffer2D(int elementSize) {
        this.elementSize = elementSize;
    }

    /**
     * Allocates two new 2D buffers in device memory. Also frees old memory, if needed.
     * @param w new width
     * @param h nw height
     */
    void reallocate(int w, int h){
        memoryFree();
        reallocatePrimary2DBuffer(w, h);
        reallocateSecondary2DBuffer(w, h);
    }

    private void reallocatePrimary2DBuffer(int w, int h){
        array1 = new CUdeviceptr();
        array1Pitch = allocateDevice2DBuffer(w, h, array1);
        setPrimary2DBufferDirty(true);
    }
    private void reallocateSecondary2DBuffer(int w, int h){
        array2 = new CUdeviceptr();
        array2Pitch = allocateDevice2DBuffer(w, h, array2);
    }

    void memoryFree(){
        if(array1 != null) {
            JCuda.cudaFree(array1);
            array1 = null;
        }
        if(array2 != null) {
            JCuda.cudaFree(array2);
            array2 = null;
        }
    }

    CUdeviceptr getPrimary2DBuffer(){
        return array1;
    }
    CUdeviceptr getSecondary2DBuffer(){
        return array2;
    }
    long getPrimary2DBufferPitch(){return array1Pitch; }
    long getSecondary2DBufferPitch(){return array2Pitch; }

    private boolean buffersSwitched = false;

    /**
     * Make the primary buffer a secondary one and vice versa
     */
    void switch2DBuffers(){
        buffersSwitched = !buffersSwitched;
        CUdeviceptr switchPtr = array1;
        array1 = array2;
        array2 = switchPtr;
        long switchPitch = array1Pitch;
        array1Pitch = array2Pitch;
        array2Pitch = switchPitch;
    }
    boolean isBuffersSwitched(){return  buffersSwitched; }

    /***
     * makes buffers order the initial one (whether the buffers have been previously switched or not)
     */
    void resetBufferOrder(){
        if(isBuffersSwitched())
            switch2DBuffers();
    }

    private boolean primary2DBufferDirty = true;

    public boolean isPrimary2DBufferDirty() {
        return primary2DBufferDirty;
    }

    public void setPrimary2DBufferDirty(boolean primary2DBufferDirty) {
        this.primary2DBufferDirty = primary2DBufferDirty;
    }

    /**
     * Allocate 2D array on device
     * Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
     *
     * @param width
     * @param height
     * @param target output parameter, will contain pointer to allocated memory
     * @return pitch (actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.)
     */
    private long allocateDevice2DBuffer(int width, int height, CUdeviceptr target) {

        /**
         * Pitch = actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.
         */
        long pitch;
        long[] pitchptr = new long[1];

        if (target != null) {
            JCuda.cudaFree(target);
        } else {
            target = new CUdeviceptr();
        }

        if (width * height == 0)
            return 0;

        //JCuda.cudaMallocPitch(target, pitchptr, (long) width * (long) Sizeof.INT, (long) height);
        JCudaDriver.cuMemAllocPitch(target, pitchptr, (long) width * (long) Sizeof.INT * elementSize, (long) height, Sizeof.INT * elementSize);

        pitch = pitchptr[0];
        if (pitch <= 0) {
            throw new CudaException("cuMemAllocPitch returned pitch with value 0 (or less)");
        }

        if (pitch > Integer.MAX_VALUE) {
            //this would mean an array with length bigger that Integer.MAX_VALUE and this is not supported by Java
            System.err.println("Warning: allocateDevice2DBuffer: pitch > Integer.MAX_VALUE");
        }
        return pitch;
    }

    @Override
    public void close() {
        memoryFree();
    }
}
