import ml.convnet.ConvNet;
import ml.convnet.layer.*;
import ml.convnet.layer.activation.*;
import ml.convnet.layer.loss.*;
import ml.convnet.trainer.*;
import ml.data.Example;

public class Test {
	public static void main(String[] args) {

		ConvNet net = new ConvNet();

		net.addLayer(new Input(1, 1, 2));
		net.addLayer(new FullConnect(10, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new FullConnect(1, 1.0));
		net.addLayer(new Sigmoid());
		net.addLayer(new Regression());

		net.trainer(new SGDTrainer(0.5, 1, 0, 0, 0));
		
		Example[] d = new Example[4];
		d[0] = new Example(new double[] {0,0}, new double[] {0});
		d[1] = new Example(new double[] {0,1}, new double[] {0});
		d[2] = new Example(new double[] {1,0}, new double[] {0});
		d[3] = new Example(new double[] {1,1}, new double[] {1});
		
		net.epoch = 50;
		net.train(d);

		int[] a = new int[10];
		
		for (int i = 0; i < a.length; i++) {
			a[i] = i;
		}
		
		for (int i = a.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			int t = a[i];
			a[i] = a[j];
			a[j] = t;
		}
		
		for (int i = 0; i < a.length; i++) {
			System.out.println(a[i]);
		}

	}
}
