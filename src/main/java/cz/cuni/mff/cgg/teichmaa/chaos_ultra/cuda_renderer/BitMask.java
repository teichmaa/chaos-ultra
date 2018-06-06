package cz.cuni.mff.cgg.teichmaa.chaos_ultra.cuda_renderer;

public class BitMask {
    private int value = 0;
    public void setBit(int n, boolean value){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        if(value)
            setBit(n);
        else
            clearBit(n);
    }
    public void setBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        value |= 1 << n;
    }
    public void clearBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        value &= ~(1 << n);
    }
    public boolean getBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        return ((value >> n) & 1) != 0;
    }
    public int getValue(){
        return value;
    }

    @Override
    public String toString() {
        return Integer.toBinaryString(value);
    }
}
