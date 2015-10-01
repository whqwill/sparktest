import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by hwang on 01.10.15.
 */
public class Test {
    public static void main(String[] args) {
        INDArray a = Nd4j.create(3,5);
        INDArray b = Nd4j.create(5,3);
        System.out.println(a.mmul(b));
        System.out.println(b.mmul(a));
    }
}
