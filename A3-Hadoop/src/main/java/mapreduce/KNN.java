package mapreduce;

import java.util.Date;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
// import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.NLineInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KNN {

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("test.dataset.path", args[1]);
        conf.setInt("k", Integer.parseInt(args[2]));

        Job job = Job.getInstance(conf, "K-Nearest Neighbors");

        job.setJarByClass(KNN.class);

        job.setInputFormatClass(NLineInputFormat.class);
        NLineInputFormat.addInputPath(job, new Path(args[0]));
        job.getConfiguration().setInt("mapreduce.input.lineinputformat.linespermap", 750);

        // input and output files
        // FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        // map and reducer
        job.setMapperClass(KNNMapper.class);
        job.setReducerClass(KNNReducer.class);

        // input and output value of Mapper
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(CDWritable.class);

        // input and output value of Reducer (and Mapper also if no setMapOutput
        // functions are called)
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);
        long start = new Date().getTime();
        boolean success = job.waitForCompletion(true);
        long end = new Date().getTime();
        System.out.println("Job took " + (end - start) + "ms");
        System.exit(success ? 0 : 1);
    }
}
