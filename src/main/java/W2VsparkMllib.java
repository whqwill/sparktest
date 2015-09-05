import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by will on 05/09/15.
 */
public class W2VsparkMllib {
    private static Logger log = LoggerFactory.getLogger(W2VsparkMllib.class);

    public static void main(String[] args) throws Exception {
        long begintime = System.currentTimeMillis();
        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("sparktest");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

    }
}