package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import jcuda.CudaException;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

import java.io.Closeable;

class DeviceMemory implements Closeable {

    private CUdeviceptr output2DArray1 = new CUdeviceptr();
    private CUdeviceptr output2DArray2 = new CUdeviceptr();
    private long output2DArray1Pitch;
    private long output2DArray2Pitch;
    private CUdeviceptr randomValues;

    /**
     * Reallocate the 2D buffer and internally mark is as empty
     * @param w
     * @param h
     */
    void reallocatePrimary2DBuffer(int w, int h){
        output2DArray1Pitch = allocateDevice2DBuffer(w, h, output2DArray1);
        setPrimary2DBufferUnusable(true);
    }
    void reallocateSecondary2DBuffer(int w, int h){
        output2DArray2Pitch = allocateDevice2DBuffer(w, h, output2DArray2);
    }

    CUdeviceptr getPrimary2DBuffer(){
        return output2DArray1;
    }
    CUdeviceptr getSecondary2DBuffer(){
        return output2DArray2;
    }
    long getPrimary2DBufferPitch(){return output2DArray1Pitch; }
    long getSecondary2DBufferPitch(){return output2DArray2Pitch; }

    boolean buffersSwitched = false;
    void switch2DBuffers(){
        buffersSwitched = !buffersSwitched;
        CUdeviceptr switchPtr = output2DArray1;
        output2DArray1 = output2DArray2;
        output2DArray2 = switchPtr;
        long switchPitch = output2DArray1Pitch;
        output2DArray1Pitch = output2DArray2Pitch;
        output2DArray2Pitch = switchPitch;
    }
    boolean isBuffersSwitched(){return  buffersSwitched; }

    /***
     * returns buffers to initial order, whether they have been previously switched or not
     */
    void resetBufferSwitch(){
        if(isBuffersSwitched())
            switch2DBuffers();
    }

    private boolean primary2DBufferUnusable = true;

    public boolean isPrimary2DBufferUnusable() {
        return primary2DBufferUnusable;
    }

    public void setPrimary2DBufferUnusable(boolean primary2DBufferUnusable) {
        this.primary2DBufferUnusable = primary2DBufferUnusable;
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
        JCudaDriver.cuMemAllocPitch(target, pitchptr, (long) width * (long) Sizeof.INT, (long) height, Sizeof.INT);

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
        if(output2DArray1 != null)
            JCuda.cudaFree(output2DArray1);
        if(output2DArray2 != null)
            JCuda.cudaFree(output2DArray2);
        if (randomValues != null)
            JCuda.cudaFree(randomValues);
    }
}
