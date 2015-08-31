/**
 * Created by hwang on 8/31/15.
 */
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import java.util.Arrays;
import java.util.List;
public class testRDD {
    public static void main(String[] args) throws Exception {
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[2]")
                .setAppName("sparktest");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        System.out.println(data);
        JavaRDD<Integer> distData = sc.parallelize(data);

        System.out.println(distData.count());
        System.out.println("finish!");
    }
}
