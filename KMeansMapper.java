import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.Arrays;

public class KMeansMapper extends Mapper<LongWritable, Text, Text, ClusterWritable> {
    private static final double EPSILON = 0.001;
    private static final int MAX_ITERATIONS = 100;

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] line = value.toString().split(",");
        double[] point = new double[line.length - 1];
        for (int i = 0; i < point.length; i++) {
            point[i] = Double.parseDouble(line[i]);
        }

        ClusterWritable closestCluster = findClosestCluster(point);
        Text outputKey = new Text(closestCluster.getCenter().toString());
        context.write(outputKey, closestCluster);
    }

    private ClusterWritable findClosestCluster(double[] point) {
        ClusterWritable closestCluster = null;
        double minDistance = Double.MAX_VALUE;

        for (ClusterWritable cluster : clusters) {
            double distance = euclideanDistance(point, cluster.getCenter().getPoint());
            if (distance < minDistance) {
                minDistance = distance;
                closestCluster = cluster;
            }
        }

        return closestCluster;
    }

    private double euclideanDistance(double[] point1, double[] point2) {
        double sum = 0.0;
        for (int i = 0; i < point1.length; i++) {
            sum += Math.pow(point1[i] - point2[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static class ClusterWritable implements Writable {
        private Cluster cluster;

        public ClusterWritable() {
        }

        public ClusterWritable(Cluster cluster) {
            this.cluster = cluster;
        }

        public Cluster getCenter() {
            return cluster;
        }

        @Override
        public void write(DataOutput out) throws IOException {
            cluster.getPoint().write(out);
            out.writeDouble(cluster.getSumOfSquaredDistances());
            out.writeInt(cluster.getNumberOfPoints());
        }

        @Override
        public void readFields(DataInput in) throws IOException {
            double[] point = new double[cluster.getPoint().length];
            for (int i = 0; i < point.length; i++) {
                point[i] = in.readDouble();
            }
            double sumOfSquaredDistances = in.readDouble();
            int numberOfPoints = in.readInt();

            cluster = new Cluster(point, sumOfSquaredDistances, numberOfPoints);
        }
    }
}