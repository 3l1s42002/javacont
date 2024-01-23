import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KMeansReducer extends Reducer<Text, KMeansMapper.ClusterWritable, Text, NullWritable> {
    private static final double EPSILON = 0.001;
    private static final int MAX_ITERATIONS = 100;

    @Override
    protected void reduce(Text key, Iterable<KMeansMapper.ClusterWritable> values, Context context) throws IOException, InterruptedException {
        List<double[]> points = new ArrayList<>();
        double sumOfSquaredDistances = 0.0;
        int numberOfPoints = 0;

        for (KMeansMapper.ClusterWritable value : values) {
            points.add(value.getCenter().getPoint());
            sumOfSquaredDistances += value.getCenter().getSumOfSquaredDistances();
            numberOfPoints += value.getCenter().getNumberOfPoints();
        }

        double[] newCenter = calculateNewCenter(points, sumOfSquaredDistances, numberOfPoints);

        if (Arrays.equals(newCenter, getCenter(context).getPoint()) || numberOfIterations >= MAX_ITERATIONS) {
            context.getCounter(KMeansCounters.CONVERGED).increment(1);
            context.write(key, NullWritable.get());
        } else {
            context.getCounter(KMeansCounters.ITERATIONS).increment(1);
            setCenter(context, new Cluster(newCenter, sumOfSquaredDistances, numberOfPoints));
            context.write(key, NullWritable.get());
        }
    }

    private double[] calculateNewCenter(List<double[]> points, double sumOfSquaredDistances, int numberOfPoints) {
        double[] newCenter = new double[points.get(0).length];
        for (double[] point : points) {
            for (int i = 0; i < point.length; i++) {
                newCenter[i] += point[i];
            }
        }
        for (int i = 0; i < newCenter.length; i++) {
            newCenter[i] /= numberOfPoints;
        }
        return newCenter;
    }

    private Cluster getCenter(Context context) throws IOException {
        return (Cluster) context.getCacheFiles().get(0).getFileSystem(context.getConfiguration()).open(new Path(context.getCacheFiles().get(0).getPath()));
    }

    private void setCenter(Context context, Cluster newCenter) throws IOException {
        context.getCounter(KMeansCounters.WRITTEN_CENTERS).increment(1);
        context.getCounter(KMeansCounters.WRITTEN_BYTES).add(