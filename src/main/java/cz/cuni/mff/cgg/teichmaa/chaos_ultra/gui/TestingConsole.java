package cz.cuni.mff.cgg.teichmaa.chaos_ultra.gui;

import cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer.AssertException;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class TestingConsole {
    public static void main(String[] args) {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
        String s = null;
        try {
            s = "";
            while (s != null) {
                System.out.print(" > ");
                s = in.readLine();
                if(s.trim().equals("")) continue;
                String[] a = s.split("\\s");
                int ss = getAdvicedSampleCount(Integer.parseInt(a[0]),Integer.parseInt(a[1]),960,540,100);
                System.out.println(ss);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static float screenDistance = 60; //in cm

    private static int getAdvicedSampleCount(int x, int y, int focusx, int focusy, int maxSuperSamplingLevel) {
        float pixelRealWidthInCm = 0.02652f; //todo this value should probably be entered by the user. From http://www.prismo.ch/comparisons/desktop.php
        float focusDistance = manhattanDistanceTo(x, y, focusx, focusy) * pixelRealWidthInCm; //distance to focus, translated to cm
        float visualAngle = 2 * (float) Math.atan(focusDistance / screenDistance) * 180/(float)Math.PI; //from https://en.wikipedia.org/wiki/Visual_angle

        if (visualAngle < 0) throw new AssertException("visualAngle <= 0: " + visualAngle);
        float fovealViewLimit = 5.5f; //in degrees, value from https://en.wikipedia.org/wiki/Peripheral_vision
        float relativeQuality = 1 / (visualAngle - fovealViewLimit + 1);  //todo it would be better to use another hyperbolic model, which reaches zero at around 70
        if (visualAngle <= fovealViewLimit) relativeQuality = 1;

        return (int) (maxSuperSamplingLevel * relativeQuality);
    }

    private static int manhattanDistanceTo(int ax, int ay, int bx, int by) {
        return Math.abs(ax - bx) + Math.abs(ay - by);
    }
}
