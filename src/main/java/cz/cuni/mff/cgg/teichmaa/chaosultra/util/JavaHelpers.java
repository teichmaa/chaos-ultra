package cz.cuni.mff.cgg.teichmaa.chaosultra.util;

public class JavaHelpers {
    public static void printCallingMethodName(){
        System.out.println(getCallingMethodName(1));
    }

    public static String getCallingMethodName(){
        return getCallingMethodName(1);
    }

    /**
     *
     * @param numberOfUpNesting how many levels to nest up: direct caller's name (0), or caller's caller's (1), or caller's caller's caller's name (2) etc.
     * @return method name of the caller
     */
    public static String getCallingMethodName(int numberOfUpNesting){
        if(numberOfUpNesting < 0 ) throw new IllegalArgumentException("must be non-negative");
        return Thread.currentThread().getStackTrace()[2 + numberOfUpNesting].getMethodName();
    }

    public static boolean isDebugMode(){
        return Boolean.toString(true).equals(System.getProperty("debug"));
    }
}
