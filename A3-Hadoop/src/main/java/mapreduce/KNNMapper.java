package mapreduce;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.util.Pair;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class KNNMapper extends Mapper<Object, Text, Text, CDWritable> {
    String testDatasetPath;
    int k;
    int numAttributes;
    long testInstances;
    ArrayList<KSmallestListPair> CD = new ArrayList<KSmallestListPair>();
    ArrayList<ArrayList<Float>> testData = new ArrayList<>();

    protected void insertInCD(int testIndex, Float newDistance, Float newClass) {
        if ((CD.size() == 0 && testIndex == 0) || CD.size() <= testIndex) {
            CD.add(new KSmallestListPair(k));
        }

        CD.get(testIndex).insertInList(newDistance, newClass);
    }

    protected ArrayList<Float> getAttributesFromLine(String line) {
        ArrayList<Float> dataInstance = new ArrayList<Float>();
        for (String word : line.split(",")) {
            dataInstance.add(Float.parseFloat(word));
        }
        return dataInstance;
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        testDatasetPath = conf.get("test.dataset.path");
        k = conf.getInt("k", 3);
        try {
            File testFile = new File(testDatasetPath);
            BufferedReader br = new BufferedReader(new FileReader(testFile));
            String line;
            while ((line = br.readLine()) != null) {
                ArrayList<Float> testInstance = getAttributesFromLine(line);
                testData.add(testInstance);
            }
            br.close();
            testInstances = testData.size();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        ArrayList<Float> trainInstance = getAttributesFromLine(value.toString());

        // try {
        // File testFile = new File(testDatasetPath);
        // BufferedReader br = new BufferedReader(new FileReader(testFile));
        // String line;
        // int i = 0;
        // while ((line = br.readLine()) != null) {
        // ArrayList<Float> testInstance = getAttributesFromLine(line);
        // int numAttributes = testInstance.size();
        // Float distance = 0f;
        // for (int j = 0; j < numAttributes - 1; j++) {
        // Float temp = trainInstance.get(j) - testInstance.get(j);
        // distance += temp * temp;
        // }
        // distance = (float) Math.sqrt(distance.doubleValue());
        // insertInCD(i++, distance, trainInstance.get(numAttributes - 1));
        // }

        // br.close();
        // } catch (IOException e) {
        // e.printStackTrace();
        // }
        for (int i = 0; i < testInstances; i++) {
            ArrayList<Float> testInstance = testData.get(i);
            int numAttributes = testInstance.size();
            Float distance = 0f;
            for (int j = 0; j < numAttributes - 1; j++) {
                Float temp = trainInstance.get(j) - testInstance.get(j);
                distance += temp * temp;
            }
            distance = (float) Math.sqrt(distance.doubleValue());
            insertInCD(i, distance, trainInstance.get(numAttributes - 1));
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        Text key = new Text(context.getTaskAttemptID().getTaskID().toString());
        CDInstanceWritable CDInstances[][] = new CDInstanceWritable[CD.size()][k];
        for (int i = 0; i < CD.size(); i++) {
            for (int j = 0; j < k; j++) {
                Pair<Float, Float> entry = CD.get(i).get(j);
                CDInstanceWritable CDInstance = new CDInstanceWritable();
                FloatWritable temp[] = new FloatWritable[2];
                temp[0] = new FloatWritable(entry.getFirst());
                temp[1] = new FloatWritable(entry.getSecond());
                CDInstance.set(temp);
                CDInstances[i][j] = CDInstance;
            }
        }
        CDWritable CDEmit = new CDWritable();
        CDEmit.set(CDInstances);
        context.write(key, CDEmit);
    }
}
