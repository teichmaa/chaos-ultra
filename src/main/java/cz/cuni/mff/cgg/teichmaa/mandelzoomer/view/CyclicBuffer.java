package cz.cuni.mff.cgg.teichmaa.mandelzoomer.view;

public class CyclicBuffer {

    private int capacity;
    private int[] data;
    private int index = 0;

    private int sum = 0;

    public CyclicBuffer(int capacity) {
        this.capacity = capacity;
        data = new int[capacity];
    }
    public CyclicBuffer(int capacity, int initialValue) {
        this(capacity);
        for (int i = 0; i < capacity; i++) {
            add(initialValue);
        }
    }

    public void add(int value){
        if(value > 2000) {
            int a = 0;
        }
        sum = sum - data[index] + value;
        data[index] = value;
        index = (index + 1) % capacity;
    }

    public int get(int index){
        if(index < 0 || index >= capacity) throw new IllegalArgumentException("index must be >= 0 and < capacity.");
        return data[index];
    }

    public float getMeanValue(){
        return sum / (float)capacity;
    }

    public int getCapacity() {
        return capacity;
    }
}
