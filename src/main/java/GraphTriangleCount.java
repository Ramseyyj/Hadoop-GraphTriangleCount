import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class GraphTriangleCount {
    /**
     * 有向图转换为无向图 input key: 行偏移 input value: 行内容，有向边from->to output key:
     * 转换为无向，即输出from->to和to->from output value: null
     */
    public static class FileMapper extends Mapper<Object, Text, Text, NullWritable> {
        NullWritable nw = NullWritable.get();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] split = value.toString().split(" ");// split即from和to的String数组
            if (!split[0].equals(split[1])) {// 消除自己指向自己的情况
                context.write(new Text(split[0] + " " + split[1]), nw);// from->to
                context.write(new Text(split[1] + " " + split[0]), nw);// to->from
            }
        }
    }

    /**
     * 重载Partitioner 同一出发顶点分配到同一reducer 使Mapper自动完成对邻接点的排序
     */
    public static class FilePartitioner extends HashPartitioner<Text, NullWritable> {
        @Override
        public int getPartition(Text key, NullWritable value, int numReduceTasks) {
            return super.getPartition(new Text(key.toString().split(" ")[0]), value, numReduceTasks);
        }
    }

    /**
     * 得到所有临接点 Reduce主要工作为去重和收集所有邻接点 input key: 临接关系 from->to
     * (不同于map的from->to，这里为无向) input value: null output key: 顶点 output value:
     * 其所有邻接点，以space隔开
     */
    public static class FileReducer extends Reducer<Text, NullWritable, Text, Text> {
        String lastKey = "";// 上次的顶点
        StringBuilder out = new StringBuilder();// 暂存此顶点的邻接点

        public void reduce(Text key, Iterable<NullWritable> values, Context context)
                throws IOException, InterruptedException {
            String[] temp = key.toString().split(" ");// 得到顶点和其一个邻接点
            if (!(temp[0].equals(lastKey) || lastKey.equals(""))) {// 如果此次得到的顶点不同于上次，则上次顶点的所有邻接点找齐，输出
                out.setLength(out.length() - 1);// 去除最后的space
                context.write(new Text(lastKey), new Text(out.toString()));// 输出
                out = new StringBuilder();// 清空out，为新的点做准备
            }
            out.append(temp[1] + " ");// 同一顶点临接点自动合并了，所以遇到的每个都可以直接加入到邻接点集合中
            lastKey = temp[0];// 更新上次顶点，这里写在if外是为了避免第一次lastkey为空造成问题
        }

        /**
         * cleanup()处理最后一次out没输出的情况
         */
        public void cleanup(Context context) throws IOException, InterruptedException {
            out.setLength(out.length() - 1);
            context.write(new Text(lastKey), new Text(out.toString()));
        }
    }

    /**
     * 得到某点，和其邻接点，和邻接点的邻接点 输入为A->(B,C,D...) 这样得到B->A->C,B->A->D...
     * 对其邻接点(B)和邻接点的邻接点(C),以B和C的大小排序后输出
     * 即如果A->B->C，且A->C->B，则输出为2个A->B->C，这样即可知道这三个点是否构成三角形
     * 对于无向图，如果ABC构成三角形，可能是A->B->C, A->C->B, B->A->C, B->C->A, C->A->B, C-B-A
     * 但ABC三者之间有大小关系，如果仅输出A->B->C (A<B<C)，则唯一确定一个三角形 所以map输出的是满足上述关系的三角形三个顶点
     * 为了减少map输出，将同一第一个顶点的所有后面两个顶点集合到一起输出 后面两个顶点之间以space分开，不同顶点集间以tab分开 input
     * key: 行偏移 input value: 顶点和其所有邻接点 output key: 第一个顶点 output valye:
     * （剩下两个顶点）组成的集合
     */
    public static class Mapper2 extends Mapper<Object, Text, Text, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tmp = value.toString().split("\t");// 得到第一个顶点，和其邻接点
            String k = tmp[0];// 第一个顶点，后为“第二个顶点”
            String[] v = tmp[1].split(" ");// 得到其邻接点集合
            int length = v.length - 1;
            for (int index = 0; index < length; index++) {// 每次将其邻接点抽一个出来，作为新的“第一个顶点”
                StringBuilder sb = new StringBuilder();// 暂存（剩下的两个顶点）的集合
                String first = v[index];// 新的“第一个顶点”
                if (first.compareTo(k) > 0) {// 如果第一个顶点比第二个顶点（即原“第一个顶点”）值大，则不满足要求，因邻接点已经排序，其后的顶点必然不满足要求
                    break;
                }
                for (int i = index + 1; i <= length; i++) {// 便利邻接点，作为“第三个顶点”
                    String now = v[i];// 第三个顶点
                    if (k.compareTo(now) < 0) {// 如果第二个顶点比第三个顶点小
                        sb.append(k + " " + now + "\t");
                    } else {// 如果第二个顶点比第三个顶点大
                        sb.append(now + " " + k + "\t");
                    }
                }
                sb.setLength(sb.length() - 1);// 去除最后的tab
                context.write(new Text(first), new Text(sb.toString()));
            }
        }
    }

    /**
     * 收集同一个第一个顶点的情况 统计（第二个顶点和第三个顶点）的个数 统计（第二个顶点和第三个顶点）的集合的个数（去重）
     * 其差值即为重复的个数，重复情况即A->B->C, A->C->B，可以构成三角形 所以重复个数即构成三角形个数
     * 因为题目要求，不需要知道具体哪三个点构成三角形，只需要输出个数 input value: 第一个顶点 input key:
     * 多个（第二个顶点和第三个顶点）集合 output key: 此reducer得到的所有三角形个数 output value: null
     */
    public static class Reducer2 extends Reducer<Text, Text, IntWritable, NullWritable> {
        private int count = 0;// 统计此reducer总共得到的三角形个数，区别于单次

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int c = 0;// 所有（第二个顶点和第三个顶点）出现个数（重复情况）
            Set<String> set = new HashSet<String>();// 设置set（去重）
            for (Text ta : values) {// 每个集合
                for (String s : ta.toString().split("\t")) {// 每个（第二个顶点和第三个顶点）
                    set.add(s);// 加入到set（去重）
                    c++;
                }
            }
            count += (c - set.size());// c-set.size()即此次得到的三角形，加到count统计此reducer得到的所有三角形
        }

        /**
         * 最后将统计到的三角形个数输出
         */
        public void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new IntWritable(count), NullWritable.get());
        }
    }

    /**
     * 上一步得到了所有三角形个数，但分散在多个reduce文件中 这时再增加一个job，将三角形个数汇总到一个文件 input key: 行偏移
     * input value: 上一步每个reducer得到的三角形个数 output key: same,
     * 无关紧要，设为相同值能在reduce中一次相加完成 output value: 上一步每个reducer得到的三角形个数
     * （实际上是为了满足题目要求“输出三角形个数到一个文件”而增加的一个job, 这样简单的相加用MapReduce需要耗时约40秒
     * 但如果用一般脚本语言，大概0.1秒就能完成...）
     */
    public static class CountMapper extends Mapper<Object, Text, Text, IntWritable> {
        Text same = new Text("same");// key值，无关紧要，但保持相同

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            context.write(same, new IntWritable(Integer.parseInt(value.toString())));
        }
    }

    public static class CountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        int count = 0;// 统计所有三角形个数

        public void reduce(Text key, Iterable<IntWritable> values, Context context) {
            for (IntWritable i : values) {// 累加三角形个数
                count += i.get();
            }
        }

        /**
         * 最后输出总的三角形个数
         */
        public void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("triangle count"), new IntWritable(count));
        }
    }

    public static void main(String[] args) throws Exception {
        /*
         * 执行时有4个命令行参数 input tmp1 tmp2 output input即输入文件路径，在这里是有向边关系
		 * tmp1即job1输出文件路径，即点和其临接点 tmp2即jib2输出文件路径，即每个reducer统计到的三角形个数
		 * output即job3输出文件路径，即总的三角形个数
		 */
        // job1
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        Job job = new Job(conf, "Triangle");
        job.setJarByClass(GraphTriangleCount.class);
        job.setMapperClass(FileMapper.class);
        job.setPartitionerClass(FilePartitioner.class);
        job.setReducerClass(FileReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(NullWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(50);// 设定reducer数目
        FileInputFormat.setMinInputSplitSize(job, 1);// 设定文件分片，这样才能让多个mapper和reducer实际用起来
        FileInputFormat.setMaxInputSplitSize(job, 10485760);
        FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
        job.waitForCompletion(true);
        // job2
        Configuration conf2 = new Configuration();
        Job job2 = new Job(conf2, "Triangle2");
        job2.setJarByClass(GraphTriangleCount.class);
        job2.setMapperClass(Mapper2.class);
        job2.setReducerClass(Reducer2.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(Text.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);
        job2.setNumReduceTasks(50);// 设定reducer数目
        FileInputFormat.setInputPaths(job2, new Path(otherArgs[1]));
        FileOutputFormat.setOutputPath(job2, new Path(otherArgs[2]));
        job2.waitForCompletion(true);// 等待前一个job完成
        // job3
        // 不设定reducer数目因为必须只能有1个reducer
        Configuration conf3 = new Configuration();
        Job job3 = new Job(conf3, "Triangle3");
        job3.setJarByClass(GraphTriangleCount.class);
        job3.setMapperClass(CountMapper.class);
        job3.setReducerClass(CountReducer.class);
        job3.setMapOutputKeyClass(Text.class);
        job3.setMapOutputValueClass(IntWritable.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(IntWritable.class);
        FileInputFormat.setInputPaths(job3, new Path(otherArgs[2]));
        FileOutputFormat.setOutputPath(job3, new Path(otherArgs[3]));
        System.exit(job3.waitForCompletion(true) ? 0 : 1);// 等待前一个job完成
    }
}
