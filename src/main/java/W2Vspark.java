/**
 * Created by hwang on 8/31/15.
 */
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.springframework.core.io.ClassPathResource;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Collection;

public class W2Vspark {
    private static Logger log = LoggerFactory.getLogger(W2V.class);

    public static void main(String[] args) throws Exception {
        long begintime = System.currentTimeMillis();

        System.out.println("step 1...");
        // These are all default values for word2vec
        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("sparktest");

        System.out.println("step 2...");
        // Set SparkContext
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        System.out.println("step 3...");
        // Path of data
        String dataPath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
    //        String dataPath = new ClassPathResource("spark_word2vec_test.txt").getFile().getAbsolutePath();

        System.out.println("step 4...");
        System.out.println(dataPath);
        // Read in data
        JavaRDD<String> corpus = sc.textFile(dataPath);

        System.out.println("step 5...");
        Word2Vec word2Vec = new Word2Vec()
                .setnGrams(1)
                .setTokenizer("org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
                .setTokenPreprocessor("org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
                .setRemoveStop(false)
                .setSeed(42L)
                .setNegative(0)
                .setUseAdaGrad(false)
                .setVectorLength(100)
                .setWindow(5)
                .setAlpha(0.025).setMinAlpha(0.01)
                .setIterations(1)
                .setNumWords(5);

        System.out.println(word2Vec.getNumWords());
        System.out.println("step 6...");
        word2Vec.train(corpus);

        System.out.println(word2Vec.getNumWords());
        System.out.println("step 7...");
        Collection<String> words = word2Vec.wordsNearest("day", 10);
        System.out.println(words);

        sc.stop();

        long endtime=System.currentTimeMillis();
        long costTime = (endtime - begintime);
        System.out.println("costTime:" + String.valueOf(costTime / 1000.0) + "s");
    }
}
