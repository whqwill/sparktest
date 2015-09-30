package Spark;

import net.didion.jwnl.data.Word;
import org.apache.commons.math3.util.FastMath;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
//import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
//import org.deeplearning4j.spark.models.embeddings.word2vec.FirstIterationFunction;
import org.deeplearning4j.spark.models.embeddings.word2vec.MapToPairFunction;
import org.deeplearning4j.spark.text.functions.CountCumSum;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Spark version of word2vec
 *
 * @author Adam Gibson
 */
public class Word2Vec extends WordVectorsImpl implements Serializable  {

    private INDArray trainedSyn1;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int MAX_EXP = 6;
    private double[] expTable;

    // Input by user only via setters
    private int vectorLength = 100;
    private boolean useAdaGrad = false;
    private int negative = 0;
    private int numWords = 1;
    private int window = 5;
    private double alpha= 0.025;
    private double minAlpha = 0.0001;
    private int numPartitions = 1;
    private int iterations = 1;
    private int nGrams = 1;
    private String tokenizer = "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory";
    private String tokenPreprocessor = "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor";
    private boolean removeStop = false;
    private long seed = 42L;
    private int K = 1;
    private VocabCache vocab;
    private INDArray syn0;

    // Constructor to take InMemoryLookupCache table from an already trained model
    public Word2Vec(INDArray trainedSyn1) {
        this.trainedSyn1 = trainedSyn1;
        this.expTable = initExpTable();
    }

    public Word2Vec() {
        this.expTable = initExpTable();
    }

    public double[] initExpTable() {
        double[] expTable = new double[1000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
        return expTable;
    }

    public Map<String, Object> getTokenizerVarMap() {
        return new HashMap<String, Object>() {{
            put("numWords", numWords);
            put("nGrams", nGrams);
            put("tokenizer", tokenizer);
            put("tokenPreprocessor", tokenPreprocessor);
            put("removeStop", removeStop);
        }};
    }

    public Map<String, Object> getWord2vecVarMap() {
        return new HashMap<String, Object>() {{
            put("vectorLength", vectorLength);
            put("useAdaGrad", useAdaGrad);
            put("negative", negative);
            put("window", window);
            put("alpha", alpha);
            put("minAlpha", minAlpha);
            put("iterations", iterations);
            put("seed", seed);
            put("maxExp", MAX_EXP);
            put("K", K);
        }};
    }

    // Training word2vec based on corpus
    public void train(JavaRDD<String> corpusRDD) throws Exception {
        log.info("Start training ...");

        // SparkContext
        final JavaSparkContext sc = new JavaSparkContext(corpusRDD.context());

        // Pre-defined variables
        Map<String, Object> tokenizerVarMap = getTokenizerVarMap();
        Map<String, Object> word2vecVarMap = getWord2vecVarMap();

        // Variables to fill in in train
        //final JavaRDD<AtomicLong> sentenceWordsCountRDD;
        final JavaRDD<List<VocabWord>> vocabWordListRDD;
        //final JavaPairRDD<List<VocabWord>, Long> vocabWordListSentenceCumSumRDD;
        final VocabCache vocabCache;
        final JavaRDD<Long> sentenceCumSumCountRDD;

        // Start Training //
        //////////////////////////////////////
        log.info("Tokenization and building VocabCache ...");
        // Processing every sentence and make a VocabCache which gets fed into a LookupCache
        Broadcast<Map<String, Object>> broadcastTokenizerVarMap = sc.broadcast(tokenizerVarMap);
        TextPipeline pipeline = new TextPipeline(corpusRDD.repartition(numPartitions), broadcastTokenizerVarMap);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();

        // Get total word count and put into word2vec variable map
        word2vecVarMap.put("totalWordCount", pipeline.getTotalWordCount()/numPartitions);

        // 2 RDDs: (vocab words list) and (sentence Count).Already cached
        //sentenceWordsCountRDD = pipeline.getSentenceCountRDD();
        vocabWordListRDD = pipeline.getVocabWordListRDD();

        // Get vocabCache and broad-casted vocabCache
        Broadcast<VocabCache> vocabCacheBroadcast = pipeline.getBroadCastVocabCache();
        vocabCache = vocabCacheBroadcast.getValue();

        //////////////////////////////////////
        log.info("Building Huffman Tree ...");
        // Building Huffman Tree would update the code and point in each of the vocabWord in vocabCache
        Huffman huffman = new Huffman(vocabCache.vocabWords());
        huffman.build();

        /////////////////////////////////////
        log.info("Training word2vec sentences ...");

        word2vecVarMap.put("vecNum", vocabCache.numWords());

        //Map<Tuple2<Integer,Integer>, INDArray> s0 = new HashMap();
        Map<Pair<Integer,Integer>, INDArray> s0 = new HashMap();
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < vocabCache.numWords(); i++) {
                s0.put(new Pair(i, k), getRandomSyn0Vec(vectorLength));
            }
        }
        for (int i = vocabCache.numWords(); i < vocabCache.numWords()*2-1; i++) {
            s0.put(new Pair(i,0), Nd4j.zeros(1, vectorLength));
        }

        for (int i = 0; i < iterations; i++) {
            System.out.println("iteration: "+i);

            word2vecVarMap.put("alpha", alpha-(alpha-minAlpha)/iterations*i);
            word2vecVarMap.put("minAlpha", alpha-(alpha-minAlpha)/iterations*(i+1));

            FlatMapFunction firstIterationFunction = new FirstIterationFunction(word2vecVarMap, expTable, sc.broadcast(s0));

            class MapPairFunction implements PairFunction<Map.Entry<Integer, INDArray>, Integer, INDArray> {
                public Tuple2<Integer, INDArray> call(Map.Entry<Integer, INDArray> pair) {
                    return new Tuple2(pair.getKey(), pair.getValue());
                }
            }

            class Sum implements Function2<INDArray, INDArray, INDArray> {
                public INDArray call(INDArray a, INDArray b) {
                    return a.add(b);
                }
            }

            //@SuppressWarnings("unchecked")
            JavaPairRDD<Pair<Integer,Integer>, INDArray> indexSyn0UpdateEntryRDD =
                    vocabWordListRDD.mapPartitions(firstIterationFunction).mapToPair(new MapPairFunction()).cache();
            Map<Pair<Integer,Integer>, Object> count = indexSyn0UpdateEntryRDD.countByKey();
            indexSyn0UpdateEntryRDD = indexSyn0UpdateEntryRDD.reduceByKey(new Sum());

            // Get all the syn0 updates into a list in driver
            List<Tuple2<Pair<Integer,Integer>, INDArray>> syn0UpdateEntries = indexSyn0UpdateEntryRDD.collect();

            // Updating syn0
            s0 = new HashMap();
            for (Tuple2<Pair<Integer,Integer>, INDArray> syn0UpdateEntry : syn0UpdateEntries) {
                int cc = Integer.parseInt(count.get(syn0UpdateEntry._1).toString());
                //int cc = 1;
                if (cc > 0) {
                    INDArray tmp = Nd4j.zeros(1, vectorLength).addi(syn0UpdateEntry._2).divi(cc);
                    s0.put(syn0UpdateEntry._1, tmp);
                }
            }
        }

        syn0 = Nd4j.zeros(vocabCache.numWords()*K, vectorLength);
        for (Map.Entry<Pair<Integer,Integer>, INDArray> ss: s0.entrySet()) {
            if (ss.getKey().getFirst() < vocabCache.numWords()) {
                syn0.getRow(ss.getKey().getSecond()*vocabCache.numWords()+ss.getKey().getFirst()).addi(ss.getValue());
            }
        }

        vocab = vocabCache;
        //InMemoryLookupTable inMemoryLookupTable = new InMemoryLookupTable();
        //inMemoryLookupTable.setVocab(vocabCache);
        //inMemoryLookupTable.setVectorLength(vectorLength);
        //inMemoryLookupTable.setSyn0(syn0);
        //lookupTable = inMemoryLookupTable;
    }

    public Collection<String> wordsNearest(String word, int k, int n) {
        INDArray vector = Transforms.unitVec(getWordVectorMatrix(word, k));
        

    }

    public double similarity(String word, int k1, String word2, int k2) {
        if (k1 > K || k2 > K)
            return -1;

        if(word.equals(word2) && k1 == k2)
            return 1.0;

        INDArray vector = Transforms.unitVec(getWordVectorMatrix(word, k1));
        INDArray vector2 = Transforms.unitVec(getWordVectorMatrix(word2, k2));
        if(vector == null || vector2 == null)
            return -1;
        return  Nd4j.getBlasWrapper().dot(vector, vector2);
    }

    public INDArray getWordVectorMatrix(String word, int k) {
        if(word == null || k > K)
            return null;
        int idx = vocab.indexOf(word);
        if(idx < 0)
            idx = vocab.indexOf(org.deeplearning4j.models.word2vec.Word2Vec.UNK);
        return syn0.getRow(vocab.numWords()*k+idx);
    }

    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }

    public int getVectorLength() {
        return vectorLength;
    }

    public Word2Vec setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
        return this;
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public Word2Vec setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
        return this;
    }

    public int getNegative() {
        return negative;
    }

    public Word2Vec setNegative(int negative) {
        this.negative = negative;
        return this;
    }

    public int getNumWords() {
        return numWords;
    }

    public Word2Vec setNumWords(int numWords) {
        this.numWords = numWords;
        return this;
    }

    public int getWindow() {
        return window;
    }

    public Word2Vec setWindow(int window) {
        this.window = window;
        return this;
    }

    public double getAlpha() {
        return alpha;
    }

    public Word2Vec setAlpha(double alpha) {
        this.alpha = alpha;
        return this;
    }

    public Word2Vec setNumPartitions(int numPartitions) {
        this.numPartitions = numPartitions;
        return this;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public Word2Vec setMinAlpha(double minAlpha) {
        this.minAlpha = minAlpha;
        return this;
    }

    public int getIterations() {
        return iterations;
    }

    public Word2Vec setIterations(int iterations) {
        this.iterations = iterations;
        return this;
    }

    public int getnGrams() {
        return nGrams;
    }

    public Word2Vec setnGrams(int nGrams) {
        this.nGrams = nGrams;
        return this;
    }

    public String getTokenizer() {
        return tokenizer;
    }

    public Word2Vec setTokenizer(String tokenizer) {
        this.tokenizer = tokenizer;
        return this;
    }

    public String getTokenPreprocessor() {
        return tokenPreprocessor;
    }

    public Word2Vec setTokenPreprocessor(String tokenPreprocessor) {
        this.tokenPreprocessor = tokenPreprocessor;
        return this;
    }

    public boolean isRemoveStop() {
        return removeStop;
    }

    public Word2Vec setRemoveStop(boolean removeStop) {
        this.removeStop = removeStop;
        return this;
    }

    public long getSeed() {
        return seed;
    }

    public Word2Vec setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    public double[] getExpTable() {
        return expTable;
    }
}