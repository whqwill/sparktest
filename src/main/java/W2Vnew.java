/**
 * Created by hwang on 8/31/15.
 */


import org.deeplearning4j.models.embeddings.WeightLookupTable;
//import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
//import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Collection;

public class W2Vnew {
    private static Logger log = LoggerFactory.getLogger(W2Vnew.class);

    public static void main(String[] args) throws Exception {
        long begintime = System.currentTimeMillis();

        String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache();
        InMemoryLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5).iterations(1)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache).seed(42)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");
        // Write word
        //WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");


        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println(lst);
        System.out.println(vec.similarity("day", "year"));
        System.out.println(vec.similarity("day","should"));
        long endtime=System.currentTimeMillis();
        long costTime = (endtime - begintime);
        System.out.println("costTime:"+String.valueOf(costTime/1000.0)+"s");
    }
}
