import Spark.*;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by hwang on 06.10.15.
 */
public class testLookupTable {
    public static class Pair {
        int x = 0;
        int y = 0;
        public Pair(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
    public static void addX(Pair p) {
        p.x++;
    }
    public static void createNewPair(Pair p) {
        p = new Pair(4,4);
    }
    public static void main(String[] args) throws IOException{
        Pair a = new Pair(3,3);
        addX(a);
        createNewPair(a);
        System.out.println(a.x + " " + a.y);

        INDArray b = Nd4j.create(1,4);

        b.addiRowVector(Nd4j.rand(42, new int[]{1, 4}).subi(0.5).divi(10));
        b.addRowVector(Nd4j.rand(42, new int[]{1, 4}).subi(0.5).divi(10));

        double[] c = new double[10];
        List<Integer> cc = new ArrayList<Integer>();
        for (int i = 0; i < 10; i++)
            cc.add(i+1);

        b = Nd4j.create(c);
        System.out.println(b);

        InMemoryLookupCache vocab = new InMemoryLookupCache();
        INDArray syn0 = Spark.Word2Vec.readVocab(vocab, "vectors.txt", 2);

        //System.out.println(vocab.numWords());
        Collection<String> words = Spark.Word2Vec.wordsNearest(syn0,vocab,"bank",0,40,2);
        System.out.println("bank(0): "+words);
        words = Spark.Word2Vec.wordsNearest(syn0,vocab,"bank",1,40,2);
        System.out.println("bank(1): "+words);
    }
}
