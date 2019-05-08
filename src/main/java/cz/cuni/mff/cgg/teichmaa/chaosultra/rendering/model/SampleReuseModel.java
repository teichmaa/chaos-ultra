package cz.cuni.mff.cgg.teichmaa.chaosultra.rendering.model;

public interface SampleReuseModel {

    boolean isUseSampleReuse();

    void setUseSampleReuse(boolean useSampleReuse);

    void setSampleReuseCacheDirty(boolean b);

    boolean isSampleReuseCacheDirty();
}
