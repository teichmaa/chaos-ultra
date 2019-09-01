package cz.cuni.mff.cgg.teichmaa.chaosultra.cudarenderer;

public class CudaInitializationException extends RuntimeException {
    public CudaInitializationException() {
    }

    public CudaInitializationException(String message) {
        super(message);
    }

    public CudaInitializationException(String message, Throwable cause) {
        super(message, cause);
    }

    public CudaInitializationException(Throwable cause) {
        super(cause);
    }

    public CudaInitializationException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
