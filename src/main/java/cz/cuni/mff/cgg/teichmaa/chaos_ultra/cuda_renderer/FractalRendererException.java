package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

public class FractalRendererException extends RuntimeException {
    public FractalRendererException() {
    }

    public FractalRendererException(String message) {
        super(message);
    }

    public FractalRendererException(String message, Throwable cause) {
        super(message, cause);
    }

    public FractalRendererException(Throwable cause) {
        super(cause);
    }

    public FractalRendererException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
