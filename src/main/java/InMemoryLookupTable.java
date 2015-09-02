/**
 * Created by will on 02/09/15.
 */

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.plot.dropwizard.RenderApplication;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;


public class InMemoryLookupTable implements WeightLookupTable {


    protected INDArray syn0,syn1;
    protected int vectorLength;
    protected transient Random rng = Nd4j.getRandom();
    protected AtomicDouble lr = new AtomicDouble(25e-3);
    protected double[] expTable;
    protected static double MAX_EXP = 6;
    protected long seed = 123;
    //negative sampling table
    protected INDArray table,syn1Neg;
    protected boolean useAdaGrad;
    protected double negative = 0;
    protected VocabCache vocab;
    protected Map<Integer,INDArray> codes = new ConcurrentHashMap();
    private Map<Integer, INDArray> indexSyn0VecMap = new HashMap();

    public InMemoryLookupTable() {}

    public void fit() {
        Iterator<Map.Entry<Integer,INDArray>> iter = indexSyn0VecMap.entrySet().iterator();
        List<Pair<Integer, INDArray>> a = new ArrayList<Pair<Integer, INDArray>>();
        while (iter.hasNext()) {
            Map.Entry<Integer,INDArray> en = iter.next();
            a.add(new Pair<Integer, INDArray>(en.getKey(),en.getValue()));
        }
        syn0 = Nd4j.create(vocab.numWords()+1, vectorLength);
        for (Pair<Integer, INDArray> syn0UpdateEntry : a) {
            syn0.getRow(syn0UpdateEntry.getFirst()).addi(syn0UpdateEntry.getSecond());
        }

    }

    public InMemoryLookupTable(VocabCache vocab,int vectorLength,boolean useAdaGrad,double lr,Random gen,double negative) {
        this.vocab = vocab;
        this.vectorLength = vectorLength;
        this.useAdaGrad = useAdaGrad;
        this.lr.set(lr);
        this.rng = gen;
        this.negative = negative;
        initExpTable();
    }

    public double[] getExpTable() {
        return expTable;
    }

    public void setExpTable(double[] expTable) {
        this.expTable = expTable;
    }

    @Override
    public int layerSize() {
        return vectorLength;
    }

    @Override
    public void resetWeights(boolean reset) {
        if(this.rng == null)
            this.rng = Nd4j.getRandom();
        if(syn0 == null || reset) {
            syn0 = Nd4j.rand(new int[]{vocab.numWords() + 1, vectorLength}, rng).subi(0.5).divi(vectorLength);
            INDArray randUnk = Nd4j.rand(1, vectorLength, rng).subi(0.5).divi(vectorLength);
            putVector(Word2Vec.UNK, randUnk);
        }
        if(syn1 == null || reset)
            syn1 = Nd4j.create(syn0.shape());
        initNegative();
    }

    @Override
    public void plotVocab(Tsne tsne) {
        try {
            List<String> plot = new ArrayList();
            for(String s : vocab.words()) {
                plot.add(s);
            }
            tsne.plot(syn0, 2, plot);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        try {
            RenderApplication.main(null);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Render the words via tsne
     */
    @Override
    public void plotVocab() {
        Tsne tsne = new Tsne.Builder()
                .normalize(false).setFinalMomentum(0.8f)
                .setMaxIter(1000).build();
        try {
            List<String> plot = new ArrayList();
            for(String s : vocab.words()) {
                plot.add(s);
            }
            tsne.plot(syn0, 2, plot);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * @param codeIndex
     * @param code
     */
    @Override
    public void putCode(int codeIndex, INDArray code) {
        codes.put(codeIndex,code);
    }

    /**
     * Loads the co-occurrences for the given codes
     *
     * @param codes the codes to load
     * @return an ndarray of code.length by layerSize
     */
    @Override
    public INDArray loadCodes(int[] codes) {
        return syn1.getRows(codes);
    }


    protected void initNegative() {
        if(negative > 0) {
            syn1Neg = Nd4j.zeros(syn0.shape());
            makeTable(10000,0.75);
        }
    }


    protected void initExpTable() {
        expTable = new double[1000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp =   FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i]  = tmp / (tmp + 1.0);
        }
    }



    public INDArray getRandomSyn0Vec(int vectorLength) {
        return Nd4j.rand(seed, new int[]{1 ,vectorLength}).subi(0.5).divi(vectorLength);
    }

    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     * @param nextRandom next random for sampling
     */
    @Override
    public  void iterateSample(VocabWord w1, VocabWord w2,AtomicLong nextRandom,double alpha) {
        if(w2 == null || w2.getIndex() < 0 || w1.getIndex() == w2.getIndex() || w1.getWord().equals("STOP") || w2.getWord().equals("STOP") || w1.getWord().equals("UNK") || w2.getWord().equals("UNK"))
            return;
        //current word vector
        //INDArray l1 = this.syn0.slice(w2.getIndex());
        INDArray l1 = getRandomSyn0Vec(vectorLength);


        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);


        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector

            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = useAdaGrad ?  w1.getGradient(i, (1 - code - f)) : (1 - code - f) * alpha;

            if(neu1e.data().dataType() == DataBuffer.Type.FLOAT) {
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, syn1, neu1e);
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, l1, syn1);

            }

            else {
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, syn1, neu1e);
                Nd4j.getBlasWrapper().level1().axpy(syn1.length(), g, l1, syn1);

            }




        }


        int target = w1.getIndex();
        int label;
        //negative sampling
        if(negative > 0)
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0)
                    label = 1;
                else {
                    nextRandom.set(nextRandom.get() * 25214903917L + 11);
                    int idx = Math.abs((int) (nextRandom.get() >> 16) % table.length());

                    target = table.getInt(idx);
                    if (target <= 0)
                        target = (int) nextRandom.get() % (vocab.numWords() - 1) + 1;

                    if (target == w1.getIndex())
                        continue;
                    label = 0;
                }


                if(target >= syn1Neg.rows() || target < 0)
                    continue;

                double f = Nd4j.getBlasWrapper().dot(l1,syn1Neg.slice(target));
                double g;
                if (f > MAX_EXP)
                    g = useAdaGrad ? w1.getGradient(target, (label - 1)) : (label - 1) *  alpha;
                else if (f < -MAX_EXP)
                    g = label * (useAdaGrad ?  w1.getGradient(target, alpha) : alpha);
                else
                    g = useAdaGrad ? w1.getGradient(target, label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) : (label - expTable[(int)((f + MAX_EXP) * (expTable.length / MAX_EXP / 2))]) *   alpha;
                if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,neu1e,l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,neu1e,l1);

                if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
                    Nd4j.getBlasWrapper().axpy(g,syn1Neg.slice(target),l1);
                else
                    Nd4j.getBlasWrapper().axpy((float) g,syn1Neg.slice(target),l1);
            }

        if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
            Nd4j.getBlasWrapper().axpy(1.0,neu1e,l1);

        else
            Nd4j.getBlasWrapper().axpy(1.0f,neu1e,l1);

        indexSyn0VecMap.put(target, l1);
    }

    public boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public double getNegative() {
        return negative;
    }

    public void setNegative(double negative) {
        this.negative = negative;
    }

    /**
     * Iterate on the given 2 vocab words
     *
     * @param w1 the first word to iterate on
     * @param w2 the second word to iterate on
     */
    @Override
    public  void iterate(VocabWord w1, VocabWord w2) {
        if(w2.getIndex() < 0)
            return;
        //current word vector
        INDArray l1 = this.syn0.slice(w2.getIndex());

        //error for current word and context
        INDArray neu1e = Nd4j.create(vectorLength);




        double alpha = this.lr.get();

        for(int i = 0; i < w1.getCodeLength(); i++) {
            int code = w1.getCodes().get(i);
            int point = w1.getPoints().get(i);
            if(point >= syn0.rows() || point < 0)
                throw new IllegalStateException("Illegal point " + point);
            //other word vector
            INDArray syn1 = this.syn1.slice(point);


            double dot = Nd4j.getBlasWrapper().dot(l1,syn1);

            if(dot < -MAX_EXP || dot >= MAX_EXP)
                continue;


            int idx = (int) ((dot + MAX_EXP) * ((double) expTable.length / MAX_EXP / 2.0));
            if(idx >= expTable.length)
                continue;

            //score
            double f =  expTable[idx];
            //gradient
            double g = (1 - code - f) * (useAdaGrad ?  w1.getGradient(i, alpha) : alpha);

            if(syn0.data().dataType() == DataBuffer.Type.DOUBLE) {
                Nd4j.getBlasWrapper().axpy(g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy(g, l1, syn1);
            }
            else {
                Nd4j.getBlasWrapper().axpy((float) g, syn1, neu1e);
                Nd4j.getBlasWrapper().axpy((float) g, l1, syn1);
            }
        }





        if(syn0.data().dataType() == DataBuffer.Type.DOUBLE)
            Nd4j.getBlasWrapper().level1().axpy(l1.length(), 1.0,neu1e,l1);

        else
            Nd4j.getBlasWrapper().level1().axpy(l1.length(), 1.0f, neu1e, l1);






    }


    /**
     * Reset the weights of the cache
     */
    @Override
    public void resetWeights() {
        this.rng = Nd4j.getRandom();

        syn0  = Nd4j.rand(new int[]{vocab.numWords() + 1,vectorLength},rng).subi(0.5).divi(vectorLength);
//        INDArray randUnk = Nd4j.rand(1,vectorLength,rng).subi(0.5).divi(vectorLength);
//        putVector(Word2Vec.UNK,randUnk);

        syn1 = Nd4j.create(syn0.shape());
        initNegative();

    }


    protected void makeTable(int tableSize,double power) {
        int vocabSize = syn0.rows();
        table = Nd4j.create(new FloatBuffer(tableSize));
        double trainWordsPow = 0.0;
        for(String word : vocab.words()) {
            trainWordsPow += Math.pow(vocab.wordFrequency(word), power);
        }


        for(String word : vocab.words()) {
            double d1 = Math.pow(vocab.wordFrequency(word),power) / trainWordsPow;
            for(int i = 0; i < tableSize; i++) {
                int wordIdx = vocab.indexOf(word);
                if(wordIdx < 0)
                    continue;

                table.putScalar(i,wordIdx);
                double mul = i * 1.0 / (double) tableSize;
                if(mul > d1) {
                    wordIdx++;
                    String wordAtIndex = vocab.wordAtIndex(wordIdx);
                    if(wordAtIndex == null)
                        continue;
                    d1 += Math.pow(vocab.wordFrequency(wordAtIndex),power) / trainWordsPow;

                }

            }

        }

    }

    /**
     * Inserts a word vector
     *
     * @param word   the word to insert
     * @param vector the vector to insert
     */
    @Override
    public void putVector(String word, INDArray vector) {
        if(word == null)
            throw new IllegalArgumentException("No null words allowed");
        if(vector == null)
            throw new IllegalArgumentException("No null vectors allowed");
        int idx = vocab.indexOf(word);
        syn0.slice(idx).assign(vector);

    }

    public INDArray getTable() {
        return table;
    }

    public void setTable(INDArray table) {
        this.table = table;
    }

    public INDArray getSyn1Neg() {
        return syn1Neg;
    }

    public void setSyn1Neg(INDArray syn1Neg) {
        this.syn1Neg = syn1Neg;
    }

    /**
     * @param word
     * @return
     */
    @Override
    public INDArray vector(String word) {
        if(word == null)
            return null;
        int idx = vocab.indexOf(word);
        if(idx < 0)
            idx = vocab.indexOf(Word2Vec.UNK);
        return syn0.getRow(idx);
    }

    @Override
    public void setLearningRate(double lr) {
        this.lr.set(lr);
    }

    @Override
    public Iterator<INDArray> vectors() {
        return new WeightIterator();
    }


    protected  class WeightIterator implements Iterator<INDArray> {
        protected int currIndex = 0;

        @Override
        public boolean hasNext() {
            return currIndex < syn0.rows();
        }

        @Override
        public INDArray next() {
            INDArray ret = syn0.slice(currIndex);
            currIndex++;
            return ret;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    public INDArray getSyn0() {
        return syn0;
    }

    public void setSyn0(INDArray syn0) {
        this.syn0 = syn0;
    }

    public INDArray getSyn1() {
        return syn1;
    }

    public void setSyn1(INDArray syn1) {
        this.syn1 = syn1;
    }

    public int getVectorLength() {
        return vectorLength;
    }

    public void setVectorLength(int vectorLength) {
        this.vectorLength = vectorLength;
    }

    public AtomicDouble getLr() {
        return lr;
    }

    public void setLr(AtomicDouble lr) {
        this.lr = lr;
    }

    public VocabCache getVocab() {
        return vocab;
    }

    public void setVocab(VocabCache vocab) {
        this.vocab = vocab;
    }

    public Map<Integer, INDArray> getCodes() {
        return codes;
    }

    public void setCodes(Map<Integer, INDArray> codes) {
        this.codes = codes;
    }

    public static class Builder {
        protected int vectorLength = 100;
        protected boolean useAdaGrad = false;
        protected double lr = 0.025;
        protected Random gen = Nd4j.getRandom();
        protected long seed = 123;
        protected double negative = 0;
        protected VocabCache vocabCache;





        public Builder cache(VocabCache vocab) {
            this.vocabCache = vocab;
            return this;
        }

        public Builder negative(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder vectorLength(int vectorLength) {
            this.vectorLength = vectorLength;
            return this;
        }

        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }


        public Builder lr(double lr) {
            this.lr = lr;
            return this;
        }

        public Builder gen(Random gen) {
            this.gen = gen;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }



        public InMemoryLookupTable build() {
            if(vocabCache == null)
                throw new IllegalStateException("Vocab cache must be specified");

            return new InMemoryLookupTable(vocabCache,vectorLength,useAdaGrad,lr,gen,negative);
        }
    }

    @Override
    public String toString() {
        return "InMemoryLookupTable{" +
                "syn0=" + syn0 +
                ", syn1=" + syn1 +
                ", vectorLength=" + vectorLength +
                ", rng=" + rng +
                ", lr=" + lr +
                ", expTable=" + Arrays.toString(expTable) +
                ", seed=" + seed +
                ", table=" + table +
                ", syn1Neg=" + syn1Neg +
                ", useAdaGrad=" + useAdaGrad +
                ", negative=" + negative +
                ", vocab=" + vocab +
                ", codes=" + codes +
                '}';
    }
}
