package cz.cuni.mff.cgg.teichmaa.cuda;

import jcuda.NativePointerObject;

import java.lang.reflect.Field;

public class PointerHelpers {

    private static Field address;

    static {
        try {
            address = NativePointerObject.class.getDeclaredField("nativePointer");
        } catch (NoSuchFieldException e) {
            e.printStackTrace();
        }
        address.setAccessible(true);
    }

    public static void nativePointerArtihmeticHack(NativePointerObject p, long amoutToAdd){
        try {
            long v = address.getLong(p);
            address.set(p, v + amoutToAdd);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }

    }
}