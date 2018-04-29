package cz.cuni.mff.cgg.teichmaa;

public class CudaRendererException extends RuntimeException {
    public CudaRendererException() {
    }

    public CudaRendererException(String message) {
        super(message);
    }

    public CudaRendererException(String message, Throwable cause) {
        super(message, cause);
    }

    public CudaRendererException(Throwable cause) {
        super(cause);
    }

    public CudaRendererException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
