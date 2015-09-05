import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;

/**
 * Created by will on 05/09/15.
 */
public class W2VsparkMllib {
    private static Logger log = LoggerFactory.getLogger(W2VsparkMllib.class);

    public static void main(String[] args) throws Exception {
        /*long begintime = System.currentTimeMillis();
        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("sparktest");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        String dataPath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
        JavaRDD<String> corpus = sc.textFile(dataPath);
        Word2Vec vec = new Word2Vec();
        String a = "afs dfsa ssd";

        class F implements Function<String, String[]> {
            public String[] call(String s) { return new Iterable<String>(s.split(" ")); }
        }

        Word2VecModel model = vec.fit(corpus.map(new F()));

        model.findSynonyms("day",10);*/
    }
}