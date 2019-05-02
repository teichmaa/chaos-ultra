package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.rendering_params.RenderingModel;
import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import static jcuda.driver.CUresult.CUDA_ERROR_NOT_FOUND;

public abstract class CudaKernel {

    private CUfunction function;
    private String functionName;
    private CUmodule ownerModule;

    final CUfunction getFunction(){
        return function;
    }

    /**
     *
     * @param functionName Exact (mangled, case sensitive) name of the __device__ function as defined in the .ptx file.
     * @param ownerModule cuda module that contains kernel with functionName
     */
    CudaKernel(String functionName, CUmodule ownerModule) {
        this.functionName = functionName;
        this.ownerModule = ownerModule;

        //load function:
        try {
            function = new CUfunction();
            JCudaDriver.cuModuleGetFunction(function, ownerModule, functionName);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUDA_ERROR_NOT_FOUND)))
                throw new IllegalArgumentException("Function with this name not found: " + functionName, e);
            else
                throw e;
        }

    }

    protected NativePointerObject[] params = new NativePointerObject[0];

    /**
     * @return index of the added param
     */
    protected short registerParam() {
        NativePointerObject[] newParams = new NativePointerObject[params.length + 1];
        for (int i = 0; i < params.length; i++) {
            newParams[i] = params[i];
        }
        int idx = newParams.length - 1;
        newParams[idx] = null;
        params = newParams;
        return (short) idx;
    }

    /**
     * register a kernel param and set its value
     * @param value
     * @return index of the added param
     */
    protected short registerParam(double value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * register a kernel param and set its value
     * @param value
     * @return index of the added param
     */
    protected short registerParam(float value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * register a kernel param and set its value
     * @param value
     * @return index of the added param
     */
    protected short registerParam(int value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * register a kernel param and set its value
     * @param value
     * @return index of the added param
     */
    protected short registerParam(long value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }


    /**
     * @return Array of Kernel's specific parameters.
     *
     * TODO tento radek snad jiz brzy nebude pravada, a vsechny parametry budou Java-style
     * Fields to which the object presents a public index must be defined by the caller.
     */
    public final NativePointerObject[] getKernelParams(){return params;}

    public String getFunctionName() {
        return functionName;
    }

    public CUmodule getOwnerModule() {
        return ownerModule;
    }

    /**
     * Should be overridden by descendants.
     * @param model values to use
     */
    public void setParamsFromModel(RenderingModel model){
        /* nothing */
    }
}
