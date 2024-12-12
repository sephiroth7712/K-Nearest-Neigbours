package mapreduce;

import org.apache.hadoop.io.TwoDArrayWritable;

public class CDWritable extends TwoDArrayWritable {
    public CDWritable() {
        super(CDInstanceWritable.class);
    }
}
