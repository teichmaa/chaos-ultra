package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

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

    private CUdeviceptr array1 = new CUdeviceptr();
    private CUdeviceptr array2 = new CUdeviceptr();
    private long array1Pitch;
    private long array2Pitch;

    void reallocate(int w, int h){
        reallocatePrimary2DBuffer(w, h);
        reallocatePrimary2DBuffer(w, h);
    }

    /**
     * Reallocate the 2D buffer and internally mark it as empty
     * @param w
     * @param h
     */
    private void reallocatePrimary2DBuffer(int w, int h){
        array1Pitch = allocateDevice2DBuffer(w, h, 2, array1);
        setPrimary2DBufferEmpty(true);
    }
    private void reallocateSecondary2DBuffer(int w, int h){
        array2Pitch = allocateDevice2DBuffer(w, h, 2, array2);
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

    private boolean primary2DBufferEmpty = true;

    public boolean isPrimary2DBufferEmpty() {
        return primary2DBufferEmpty;
    }

    public void setPrimary2DBufferEmpty(boolean primary2DBufferEmpty) {
        this.primary2DBufferEmpty = primary2DBufferEmpty;
    }

    /**
     * Allocate 2D array on device
     * Keep in mind that cuMemAllocPitch allocates pitch x height array, i.e. the rows may be longer than width
     *
     * @param width
     * @param height
     * @param target output parameter, will contain pointer to allocated memory
     * @param elementSize how many 4-byte elements to allocate per one cell
     * @return pitch (actual row length (in bytes) as aligned by CUDA. pitch >= width * sizeof element.)
     */
    private long allocateDevice2DBuffer(int width, int height, int elementSize, CUdeviceptr target) {

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
        if(array1 != null)
            JCuda.cudaFree(array1);
        if(array2 != null)
            JCuda.cudaFree(array2);
    }
}
