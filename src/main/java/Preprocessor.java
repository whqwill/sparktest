import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.StringCleaning;

/**
 * Created by hwang on 08.09.15.
 */
public class Preprocessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        return token;
    }
}
