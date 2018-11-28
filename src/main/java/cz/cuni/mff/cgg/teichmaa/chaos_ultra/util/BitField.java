package cz.cuni.mff.cgg.teichmaa.chaos_ultra.util;

/**
 *  Stores and manipulates 32 bits of information, represented as 32 bit integer, 0-th bit being the least significant
 */
public class BitField {

    private int value;
    /**
     *   Set the n-th bit to value (0-th bit is the least significant)
     *   @param n bit to be set, between 0 and 31 (inclusive)
     * @param value value to be set, true representing 1, false representing 0
     */
    public void setBit(int n, boolean value){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        if(value)
            setBit(n);
        else
            clearBit(n);
    }
    /**
     *  Set the n-th bit to 1 (0-th bit is the least significant)
     * @param n bit to be set, between 0 and 31 (inclusive)
     */
    public void setBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        value |= 1 << n;
    }

    /**
     *   Set the n-th bit to 0 (0-th bit is the least significant)
     *   @param n bit to be set, between 0 and 31 (inclusive)
     */
    public void clearBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        value &= ~(1 << n);
    }

    /**
     *   Gets n-th bit, true representing 1, false representing 0
     *   @param n bit to be set, between 0 and 31 (inclusive), 0-th bit is the least significant
     */
    public boolean getBit(int n){
        if(n > 32 || n < 0) throw new IllegalArgumentException("bit index must be from 0 to 31: " + n);
        return ((value >> n) & 1) != 0;
    }

    /**
     *   Returns the integer representation of the bits saved, with 0 being least significant
     */
    public int getValue(){
        return value;
    }

    @Override
    public String toString() {
        return Integer.toBinaryString(value);
    }
}
