package Spark;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jblas.NDArray;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author jeffreytang
 */
public class FirstIterationFunction
        implements FlatMapFunction< Iterator<List<VocabWord>>, Entry<Pair<Integer,Integer>, INDArray> > {

    private int ithIteration = 1;
    private int vectorLength;
    private boolean useAdaGrad;
    private int negative;
    private int window;
    private double alpha;
    private double minAlpha;
    private long totalWordCount;
    private long seed;
    private int maxExp;
    private double[] expTable;
    private Broadcast<Map<Pair<Integer,Integer>, INDArray>> syn0;
    
    private Map<Pair<Integer,Integer>, INDArray> indexSyn0VecMap;
    private AtomicLong nextRandom = new AtomicLong(5);
    private int vecNum;


    public FirstIterationFunction(Map<String, Object> word2vecVarMap,
                                  double[] expTable, Broadcast<Map<Pair<Integer,Integer>, INDArray>> syn0) {

        this.expTable = expTable;
        this.vectorLength = Integer.parseInt(word2vecVarMap.get("vectorLength").toString());
        this.useAdaGrad = Boolean.getBoolean(word2vecVarMap.get("useAdaGrad").toString());
        this.negative = Integer.parseInt(word2vecVarMap.get("negative").toString());
        this.window = Integer.parseInt(word2vecVarMap.get("window").toString());
        this.alpha = Double.parseDouble(word2vecVarMap.get("alpha").toString());
        this.minAlpha = Double.parseDouble(word2vecVarMap.get("minAlpha").toString());
        this.totalWordCount = Long.parseLong(word2vecVarMap.get("totalWordCount").toString());
        this.seed = Long.parseLong(word2vecVarMap.get("seed").toString());
        this.maxExp = Integer.parseInt(word2vecVarMap.get("maxExp").toString());
        this.vecNum = Integer.parseInt(word2vecVarMap.get("vecNum").toString());
        this.syn0 = syn0;
    }

    @Override
    public Iterable<Entry<Pair<Integer,Integer>, INDArray>> call(Iterator<List<VocabWord>> iter) {
        indexSyn0VecMap = syn0.value();
        Long last = 0L;
        Long now = 0L;
        while (iter.hasNext()) {
            List<VocabWord> vocabWordsList = iter.next();
            double currentSentenceAlpha = Math.max(minAlpha,
                    alpha - (alpha - minAlpha) * (now / (double) totalWordCount));
            if (now-last > 10000) {
                System.out.println("sentenceCumSumCount: " + now + "   currentSentenceAlpha: " + currentSentenceAlpha);
                last = now;
            }
            trainSentence(vocabWordsList, currentSentenceAlpha);
            now += vocabWordsList.size();
        }
        return indexSyn0VecMap.entrySet();
    }

    public void trainSentence(List<VocabWord> vocabWordsList, double currentSentenceAlpha) {

        if (vocabWordsList != null && !vocabWordsList.isEmpty()) {
            for (int ithWordInSentence = 0; ithWordInSentence < vocabWordsList.size(); ithWordInSentence++) {
                // Random value ranging from 0 to window size
                nextRandom.set(nextRandom.get() * 25214903917L + 11);
                int b = (int) (long) this.nextRandom.get() % window;
                VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
                if (currentWord != null) {
                    skipGram(ithWordInSentence, vocabWordsList, b, currentSentenceAlpha);
                }
            }
        }
    }

    public void skipGram(int ithWordInSentence, List<VocabWord> vocabWordsList, int b, double currentSentenceAlpha) {

        VocabWord currentWord = vocabWordsList.get(ithWordInSentence);
        if (currentWord != null && !vocabWordsList.isEmpty()) {
            int end = window * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != window) {
                    int c = ithWordInSentence - window + a;
                    if (c >= 0 && c < vocabWordsList.size()) {
                        VocabWord lastWord = vocabWordsList.get(c);
                        iterateSample(currentWord, lastWord, currentSentenceAlpha);
                    }
                }
            }
        }
    }

    public void iterateSample(VocabWord w1, VocabWord w2, double currentSentenceAlpha) {

        if (w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex())
            return;

        // error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);

        // First iteration Syn0 is random numbers
        INDArray syn0Vec; //= indexSyn0VecMap.get(w2.getIndex());
        Pair<Integer,Integer> w2Index = new Pair(w2.getIndex(),0);
        if (indexSyn0VecMap.containsKey(w2Index)) {
            syn0Vec = indexSyn0VecMap.get(w2Index);
        } else {
            syn0Vec = getRandomSyn0Vec(vectorLength); // 1 row of vector length of zeros
            indexSyn0VecMap.put(w2Index, syn0Vec);
        }

        /*double score = 1;
        for (int i = 0; i < currentWord.getCodeLength(); i++) {
            int point = currentWord.getPoints().get(i)+vecNum;

            // Point to
            INDArray syn1VecCurrentIndex;// = pointSyn1VecMap.get(point);
            if (indexSyn0VecMap.containsKey(point)) {
                syn1VecCurrentIndex = indexSyn0VecMap.get(point);
            } else {
                syn1VecCurrentIndex = Nd4j.zeros(1, vectorLength); // 1 row of vector length of zeros
                indexSyn0VecMap.put(point, syn1VecCurrentIndex);
            }

            // Dot product of Syn0 and Syn1 vecs
            double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, randomSyn0Vec, syn1VecCurrentIndex);

            int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));

            if (idx >= expTable.length)
                idx = expTable.length-1;
            if (idx < 0)
                idx = 0;

            score *= expTable[idx];
        }*/


        //
        for (int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i)+vecNum;

            // Point to
            INDArray syn1Vec;// = pointSyn1VecMap.get(point);
            Pair<Integer,Integer> pointIndex = new Pair(point,0);
            if (indexSyn0VecMap.containsKey(pointIndex)) {
                syn1Vec = indexSyn0VecMap.get(pointIndex);
            } else {
                syn1Vec = Nd4j.zeros(1, vectorLength); // 1 row of vector length of zeros
                indexSyn0VecMap.put(pointIndex, syn1Vec);
            }

            // Dot product of Syn0 and Syn1 vecs
            double dot = Nd4j.getBlasWrapper().level1().dot(vectorLength, 1.0, syn0Vec, syn1Vec);

            if (dot < -maxExp || dot >= maxExp)
                continue;

            int idx = (int) ((dot + maxExp) * ((double) expTable.length / maxExp / 2.0));

            /*if (idx >= expTable.length)
                idx = expTable.length-1;
            if (idx < 0)
                idx = 0;*/

            //score
            double f = expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ? w1.getGradient(i, currentSentenceAlpha) : currentSentenceAlpha);


            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn1Vec, neu1e);
            Nd4j.getBlasWrapper().level1().axpy(vectorLength, g, syn0Vec, syn1Vec);

            indexSyn0VecMap.put(pointIndex, syn1Vec);

        }

        // Updated the Syn0 vector based on gradient. Syn0 is not random anymore.
        Nd4j.getBlasWrapper().level1().axpy(vectorLength, 1.0f, neu1e, syn0Vec);

        indexSyn0VecMap.put(w2Index, syn0Vec);

    }

    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5D).divi(vectorLength);
    }
}
