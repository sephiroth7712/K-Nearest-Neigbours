package mapreduce;

import java.util.ArrayList;

import org.apache.commons.math3.util.Pair;

public class KSmallestListPair {
    ArrayList<Pair<Float, Float>> list;
    int k;

    public KSmallestListPair(int k) {
        list = new ArrayList<>();
        this.k = k;

        for (int i = 0; i < k; i++) {
            list.add(new Pair<Float, Float>(Float.MAX_VALUE, 0f));
        }
    }

    public Pair<Float, Float> get(int index) {
        return list.get(index);
    }

    public void insertInList(Float newDistance, Float newClass) {
        for (int i = 0; i < k; i++) {
            if (newDistance < list.get(i).getFirst()) {
                for (int x = k - 2; x >= i; x--) {
                    list.set(x + 1, list.get(x));
                }
                list.set(i, new Pair<Float, Float>(newDistance, newClass));
                break;
            }
        }
    }

}
