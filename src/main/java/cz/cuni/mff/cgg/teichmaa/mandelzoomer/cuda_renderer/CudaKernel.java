package cz.cuni.mff.cgg.teichmaa.mandelzoomer.cuda_renderer;

import jcuda.CudaException;
import jcuda.NativePointerObject;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;

import static jcuda.driver.CUresult.CUDA_ERROR_NOT_FOUND;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

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
     * @param ownerModule
     */
    CudaKernel(String functionName, CUmodule ownerModule) {
        this.functionName = functionName;
        this.ownerModule = ownerModule;

        //load function:
        try {
            function = new CUfunction();
            cuModuleGetFunction(function, ownerModule, functionName);
        } catch (CudaException e) {
            if (e.getMessage().contains(CUresult.stringFor(CUDA_ERROR_NOT_FOUND)))
                throw new IllegalArgumentException("Function with this name not found: " + functionName, e);
            else
                throw e;
        }

    }


    /**
     * @return Array of Kernel's specific parameters. Fields to which the object presents a public index must be defined by the caller.
     */
    abstract public NativePointerObject[] getKernelParams();
}
