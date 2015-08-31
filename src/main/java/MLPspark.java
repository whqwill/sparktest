/**
 * Created by hwang on 8/31/15.
 */
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import java.io.File;
import java.io.FileInputStream;
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

public class MLPspark {
    private static Logger log = LoggerFactory.getLogger(MLPspark.class);

    public static void main(String[] args) throws Exception {
        long begintime = System.currentTimeMillis();

        log.info("Spark configuration....");

        SparkConf sparkConf = new SparkConf()
                .setMaster("local")
                .setAppName("sparktest");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        final int numInputs = 4;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 100;
        long seed = 6;
        int listenerFreq = iterations / 5;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(1e-3)
                .l1(0.3).regularization(true).l2(1e-3)
                .constrainGradientToUnitNorm(true)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .nIn(2).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        log.info("Load data....");
        DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);

        SparkDl4jMultiLayer master = new SparkDl4jMultiLayer(sc,conf);

        log.info("Train model....");
        DataSet iris = iter.next();
        iris.normalizeZeroMeanZeroUnitVariance();
        iris.shuffle();
        //System.out.println(iris);
        List<DataSet> next = iris.asList();
        //System.out.println(next);

        JavaRDD<DataSet> data = sc.parallelize(next);

        MultiLayerNetwork network2 = master.fitDataSet(data);

        iter.reset();

        log.info("Evaluate weights....");
        for(org.deeplearning4j.nn.api.Layer layer : network2.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        /*
        System.out.println("step 7...");
        INDArray params = network2.params();
        File writeTo = new File(UUID.randomUUID().toString());
        Nd4j.writeTxt(params, writeTo.getAbsolutePath(), ",");
        INDArray load = Nd4j.read(new FileInputStream(writeTo.getAbsolutePath()));
        System.out.println("params:");
        System.out.println(params);
        System.out.println("load:");
        System.out.println(load);
        writeTo.delete();*/

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        DataSetIterator iterTest = new IrisDataSetIterator(numSamples, numSamples);
        DataSet test = iterTest.next();
        test.normalizeZeroMeanZeroUnitVariance();
        INDArray output = network2.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

        long endtime=System.currentTimeMillis();
        long costTime = (endtime - begintime);
        System.out.println("costTime:" + String.valueOf(costTime / 1000.0) + "s");
    }
}
