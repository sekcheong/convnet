import ml.convnet.ConvNet;
import ml.convnet.layer.*;
import ml.convnet.layer.activation.*;
import ml.convnet.layer.loss.*;
import ml.convnet.trainer.*;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.image.*;
import ml.utils.Console;
import ml.utils.Format;

public class ImageClassifier {

	private static enum Category
	{
		airplane, butterfly, flower, piano, starfish, watch
	};


	private static DataSet[] loadImageDataSets(String trainDir, String tuneDir, String testDir, int imageSize) {
		ImageDataSetReader train = new ImageDataSetReader(trainDir, imageSize);
		ImageDataSetReader tune = new ImageDataSetReader(tuneDir, imageSize);
		ImageDataSetReader test = new ImageDataSetReader(testDir, imageSize);
		DataSet[] set = new DataSet[3];
		set[0] = train.readDataSet();
		set[1] = tune.readDataSet();
		set[2] = test.readDataSet();
		return set;
	}


	public static double[] maxOut(double[] v) {
		double[] out = new double[v.length];
		double max = v[0];
		int y = 0;
		for (int i = 1; i < v.length; i++) {
			if (v[i] > max) {
				y = i;
				max = v[i];
			}
		}
		out[y] = 1;
		return out;
	}
	
	
	public static boolean isEqual(double[] u, double[] v) {
		if (u.length != v.length) return false;
		for (int i = 0; i < v.length; i++) {
			if (u[i] != v[i]) return false;
		}
		return true;
	}


	private double computeErrorRate(ConvNet net, DataSet[] test) {
		return 0;
	}


	public static void main(String[] args) {
		String trainDirectory = "./data/images/trainset/";
		String tuneDirectory = "./data/images/tuneset/";
		String testDirectory = "./data/images/testset/";
		int imageSize = 32;

		long start = System.nanoTime();
		if (args.length > 5) {
			System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_folder_path> <imageSize>");
			System.exit(1);
		}
		if (args.length >= 1) {
			trainDirectory = args[0];
		}
		if (args.length >= 2) {
			tuneDirectory = args[1];
		}
		if (args.length >= 3) {
			testDirectory = args[2];
		}
		if (args.length >= 4) {
			imageSize = Integer.parseInt(args[3]);
		}

		DataSet[] dataSets = loadImageDataSets(trainDirectory, tuneDirectory, testDirectory, imageSize);

		long end = System.nanoTime() - start;

		Console.writeLine(end * 10e-10);

		Example ex = dataSets[0].get(0);
		ConvNet net = new ConvNet();

		net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

		net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new FullConnect(ex.y.depth(), 1.0));		
		net.addLayer(new Softmax());

		double eta = 0.0005;
		double alpha = 0.9;
		double lambda = 0.00008;

		Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.00005, lambda);
		//Trainer trainer = new AdamTrainer(eta, 4, alpha, 0.00005, lambda, 0.9, 0.99, 1e-8);
		trainer.onEpoch(t -> {
			return true;
		});

		trainer.onStep(t -> {
			if ((t.step() % 10 == 0)) {

			}
			return true;
		});

		net.epochs = 30;

		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

		int correct = 0;
		for (Example k : dataSets[2].examples()) {
			double[] p = net.predict(k.x.W);
			Console.writeLine("predicted: " + Format.matrix(maxOut(p), 1));
			Console.writeLine("actual   : " + Format.matrix(k.y.W, 1));
			if (isEqual(p, k.y.W)) correct++;			
		}
		
		double acc = ((double) correct) / dataSets[2].count();
		Console.writeLine("Accuracy: " + acc);

	}

}
