package mapreduce;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.FloatWritable;

public class CDInstanceWritable extends ArrayWritable {
    public CDInstanceWritable() {
        super(FloatWritable.class);
    }
}
