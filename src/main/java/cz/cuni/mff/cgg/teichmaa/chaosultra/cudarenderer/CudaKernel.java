package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

import cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model.RenderingModel;
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
     * @param functionName Exact (mangled, case sensitive) name of the __global__ function as defined in the .ptx file.
     * @param ownerModule CUDA module that contains an implementation of the __global__ method named {@code functionName}.
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
     * Registers a kernel parameter<br>
     * The order of the calls MUST be the same as the order of parameters in the CUDA source code. <br>
     * @return index of the added param, for accessing kernel parameters in an array
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
     * Registers a kernel param and sets its value. <br>
     * The order of the calls MUST be the same as the order of parameters in the CUDA source code. <br>
     * The caller is responsible for ensuring that the type of the passed argument and type of the parameter in the CUDA source code matches.
     * @param value
     * @return index of the added param, for accessing kernel parameters in an array
     */
    protected short registerParam(double value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * Registers a kernel param and sets its value. <br>
     * The order of the calls MUST be the same as the order of parameters in the CUDA source code. <br>
     * The caller is responsible for ensuring that the type of the passed argument and type of the parameter in the CUDA source code matches.
     * @param value
     * @return index of the added param, for accessing kernel parameters in an array
     */
    protected short registerParam(float value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * Registers a kernel param and sets its value. <br>
     * The order of the calls MUST be the same as the order of parameters in the CUDA source code. <br>
     * The caller is responsible for ensuring that the type of the passed argument and type of the parameter in the CUDA source code matches.
     * @param value
     * @return index of the added param, for accessing kernel parameters in an array
     */
    protected short registerParam(int value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }

    /**
     * Registers a kernel param and sets its value. <br>
     * The order of the calls MUST be the same as the order of parameters in the CUDA source code. <br>
     * The caller is responsible for ensuring that the type of the passed argument and type of the parameter in the CUDA source code matches.
     * @param value default value to set
     * @return index of the added param, for accessing kernel parameters in an array
     */
    protected short registerParam(long value){
        short i = registerParam();
        params[i] = CudaHelpers.pointerTo(value);
        return i;
    }


    /**
     * @return Array of Kernel's specific parameters.
     *
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
     * Allows setting kernel's parameters from the model.<br>
     * Should be overridden by descendants.
     * @param model values to use
     */
    public void setParamsFromModel(RenderingModel model){
        /* nothing */
    }
}
