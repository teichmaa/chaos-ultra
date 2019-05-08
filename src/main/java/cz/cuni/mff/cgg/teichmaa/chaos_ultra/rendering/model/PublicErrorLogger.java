package cz.cuni.mff.cgg.teichmaa.chaos_ultra.rendering.model;

/**
 * Allows for logging public error that are to be displayed to the user.
 */
public interface PublicErrorLogger {
    /**
     * Logs a program error that should be displayed to the user.
     * <br />
     * The displaying of the error may be done in an asynchronous way.
     *
     * @param error
     */
    void logError(String error);
}
