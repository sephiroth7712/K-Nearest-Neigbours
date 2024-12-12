package mapreduce;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public class KNNReducer extends Reducer<Text, CDWritable, IntWritable, IntWritable> {
    int k;
    ArrayList<KSmallestListPair> CD = new ArrayList<KSmallestListPair>();

    protected void insertInCD(int testIndex, Float newDistance, Float newClass) {
        if ((CD.size() == 0 && testIndex == 0) || CD.size() <= testIndex) {
            CD.add(new KSmallestListPair(k));
        }

        CD.get(testIndex).insertInList(newDistance, newClass);
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        k = conf.getInt("k", 3);
    }

    @Override
    public void reduce(Text key, Iterable<CDWritable> CDs, Context context)
            throws IOException, InterruptedException {
        for (CDWritable CD : CDs) {
            CDInstanceWritable[][] CDInstances = (CDInstanceWritable[][]) CD.toArray();
            for (int i = 0; i < CDInstances.length; i++) {
                for (int j = 0; j < k; j++) {
                    CDInstanceWritable CDInstance = CDInstances[i][j];
                    Writable temp[] = CDInstance.get();
                    FloatWritable distWritable = (FloatWritable) temp[0];
                    FloatWritable classWritable = (FloatWritable) temp[1];
                    insertInCD(i, distWritable.get(), classWritable.get());
                }
            }
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for (int i = 0; i < CD.size(); i++) {
            Map<Integer, Integer> counts = new HashMap<Integer, Integer>();
            for (int j = 0; j < k; j++) {
                Pair<Float, Float> entry = CD.get(i).get(j);
                Integer classValue = entry.getSecond().intValue();
                if (counts.containsKey(classValue)) {
                    int freq = counts.get(classValue);
                    freq++;
                    counts.put(classValue, freq);
                } else {
                    counts.put(classValue, 1);
                }
            }

            int max_count = 0, freq_class = -1;
            for (Entry<Integer, Integer> val : counts.entrySet()) {
                if (max_count < val.getValue()) {
                    freq_class = val.getKey();
                    max_count = val.getValue();
                }
            }
            context.write(new IntWritable(i), new IntWritable(freq_class));
        }
    }
}
