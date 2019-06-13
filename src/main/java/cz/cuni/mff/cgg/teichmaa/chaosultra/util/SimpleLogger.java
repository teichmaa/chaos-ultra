package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class SimpleLogger {

    private static SimpleLogger singleton = new SimpleLogger();
    private static final String debug = "debug: ";
    private static final String error = "* ERROR *: ";
    private static final String warning = "WARNING: ";
    private static final String renderingInfo = "rendering: ";

    public static SimpleLogger get() {
        return singleton;
    }

    public SimpleLogger() {
    }

    public SimpleLogger(boolean enabled, PrintStream output) {
        this.enabled = enabled;
        this.output = new PrintWriter(output);
    }

    public SimpleLogger(boolean enabled, PrintWriter output) {
        this.enabled = enabled;
        this.output = output;
    }

    private boolean enabled = true;
    PrintWriter output = new PrintWriter(System.out);

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    private String timestamp() {
        return timeFormat.format(new Date()) + ": ";
    }

    private SimpleDateFormat timeFormat = new SimpleDateFormat("dd.MM.YY HH:mm:ss");

    public void debug(String message) {
        if (enabled)
            output.println(timestamp() + debug + message);
    }

    public void logRenderingInfo(String message) {
        if (enabled)
            output.println(timestamp() + renderingInfo + message);
    }

    public void error(String message) {
        if (enabled)
            output.println(timestamp() + error + message);
    }

    public void warning(String message) {
        if (enabled)
            output.println(timestamp() + warning + message);
    }
}
