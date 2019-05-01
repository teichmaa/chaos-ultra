package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

import com.jogamp.opengl.GL2;

import java.nio.Buffer;

import static com.jogamp.opengl.GL.*;
import static com.jogamp.opengl.GL.GL_TEXTURE_2D;
import static com.jogamp.opengl.GL2ES3.GL_QUADS;
import static com.jogamp.opengl.fixedfunc.GLMatrixFunc.GL_MODELVIEW;

public class GLHelpers {
    public static void specifyTextureSize(GL2 glContext, OpenGLTexture texture) {
        specifyTextureSizeAndData(glContext, texture, null);
    }

    public static void specifyTextureSizeAndData(GL2 glContext, OpenGLTexture texture, Buffer data) {

        glContext.glMatrixMode(GL_MODELVIEW);
        glContext.glLoadIdentity();
        glContext.glEnable(texture.getTarget());

        glContext.glBindTexture(texture.getTarget(), texture.getHandle().getValue());
        {
            glContext.glTexParameteri(texture.getTarget(), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glContext.glTexParameteri(texture.getTarget(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glContext.glTexImage2D(texture.getTarget(), 0, GL_RGBA, texture.getWidth(), texture.getHeight(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            //glTexImage2D params: target, level, internalFormat, width, height, border (must 0), format, type, data (may be null)
            //documentation: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
        }
        glContext.glBindTexture(GL_TEXTURE_2D, 0);
    }

    /**
     *
     * @param glContext context to draw on
     * @param texture texture to be drawn
     */
    public static void drawRectangle(GL2 glContext, OpenGLTexture texture){
        glContext.glMatrixMode(GL_MODELVIEW);
        //gl.glPushMatrix();
        glContext.glLoadIdentity();
        glContext.glBindTexture(texture.getTarget(), texture.getHandle().getValue());
        glContext.glBegin(GL_QUADS);
        {
            //map screen quad to texture quad and make it render
            glContext.glTexCoord2f(0f, 1f);
            glContext.glVertex2f(-1f, -1f);
            glContext.glTexCoord2f(1f, 1f);
            glContext.glVertex2f(+1f, -1f);
            glContext.glTexCoord2f(1f, 0f);
            glContext.glVertex2f(+1f, +1f);
            glContext.glTexCoord2f(0f, 0f);
            glContext.glVertex2f(-1f, +1f);
        }
        glContext.glEnd();
        glContext.glBindTexture(GL_TEXTURE_2D, 0);
        //gl.glPopMatrix();
        //gl.glFinish();
    }
}
