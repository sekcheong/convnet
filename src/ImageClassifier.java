import java.util.ArrayList;
import java.util.List;

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

	private static String[] _cats = new String[] { "airplane", "butterfly", "flower", "piano", "starfish", "watch" };


	private static DataSet[] loadImageDataSets(String trainDir, String tuneDir, String testDir, int imageSize, int options) {
		ImageDataSetReader train = new ImageDataSetReader(trainDir, _cats, imageSize, options);
		ImageDataSetReader tune = new ImageDataSetReader(tuneDir, _cats, imageSize, options);
		ImageDataSetReader test = new ImageDataSetReader(testDir, _cats, imageSize, options);
		DataSet[] set = new DataSet[3];
		set[0] = train.readDataSet();
		set[1] = tune.readDataSet();
		set[2] = test.readDataSet();
		return set;
	}


	public static double[] maxOutVector(double[] v) {
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


	private static int maxOut(double[] y) {
		int maxi = 0;
		double max = y[0];
		for (int i = 0; i < y.length - 1; i++) {
			if (y[i] > max) {
				max = y[i];
				maxi = i;
			}
		}
		return maxi;
	}


	private static double computeError(ConvNet net, Example[] test) {
		int err = 0;
		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			if (p != a) {
				err++;
			}
		}
		double rate = (double) err / test.length;
		return rate;
	}


	private static void saveErrorImages(ConvNet net, Example[] test) {
		int cnt = 0;
		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			if (p != a) {
				cnt++;
				ImageUtil.saveImage(e.x, "./bin/images/" + (cnt + "_" + a + "_" + p) + ".png");
			}
		}
	}


	private static double printConfusionMatrix(ConvNet net, Example[] test) {
		int w = test[0].y.W.length;
		int[][] confusion = new int[w][w];
		int err = 0;

		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			confusion[p][a]++;
			if (p != a) err++;
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < _cats.length; i++) {
			sb.append(Format.sprintf("%10s", _cats[i]));
		}
		Console.writeLine("          " + sb.toString());

		for (int i = 0; i < confusion.length; i++) {
			sb = new StringBuffer();
			for (int j = 0; j < confusion[i].length; j++) {
				sb.append(Format.sprintf("%10d", confusion[i][j]));
			}
			Console.writeLine(Format.sprintf("%10s", _cats[i]) + sb.toString());
		}

		return ((double) err) / test.length;
	}


	private static Example[] sampleExamples(Example[] data, double frac) {
		List<Example> items = new ArrayList<Example>();
		int n = (int) (frac * data.length);
		for (Example e : data) {
			items.add(e);
		}
		Example[] ret = new Example[n];
		for (int i = 0; i < n; i++) {
			ret[i] = items.get(i);
		}
		return ret;
	}


	private static void learningCurve(Example[] train, Example[] tune, Example[] test) {

		for (int i = 10; i <= 100; i = i + 10) {

			train = sampleExamples(train, ((double) i) / 100);

			ConvNet net = new ConvNet();

			Example ex = train[0];

			net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

			net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));

			net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));

			net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
			net.addLayer(new LeRu());
			net.addLayer(new Pool(2, 2, 2, 1));

			net.addLayer(new DropOut(0.5));

			net.addLayer(new FullConnect(ex.y.depth(), 1.0));
			net.addLayer(new Softmax());

			double eta = 0.007;
			double alpha = 0.90;
			double lambda = 0.0001;

			Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.005, lambda);

			//
			// trainer.onEpoch(t -> {
			// Console.writeLine("Epoch: " + t.epoch());
			// double err = printConfusionMatrix(t.net(), dataSets[1].examples());
			// Console.writeLine("Accuracy: " + Format.sprintf("%1.8f", (1 - err)));
			// Console.writeLine("");
			// return true;
			// });

			// trainer.onStep(t -> {
			// //Console.writeLine("step: " + t.step());
			// return true;
			// });

			net.epochs = 50;

			trainer.train(net, train, tune);
			Console.writeLine("Examples: " + train.length);
			double err = printConfusionMatrix(net, test);
			Console.writeLine("");
			Console.writeLine("Test set accuracy: " + Format.sprintf("%1.8f", (1 - err)));
			Console.writeLine(i + " " + err);

		}
	}


	public static void trainAndTest(DataSet[] dataSets) {

		ConvNet net = new ConvNet();
		Example ex = dataSets[0].get(0);

		net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

		net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));
		net.addLayer(new DropOut(0.5));

		net.addLayer(new FullConnect(ex.y.depth(), 1.0));
		net.addLayer(new Softmax());

		double eta = 0.007;
		double alpha = 0.90;
		double lambda = 0.0001;

		Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.005, lambda);

		trainer.onEpoch(t -> {
			Console.writeLine("Epoch: " + t.epoch());
			double err = printConfusionMatrix(t.net(), dataSets[1].examples());
			Console.writeLine("Tune set accuracy: " + Format.sprintf("%1.8f", (1 - err)));
			Console.writeLine("");
			if (err < 0.21) return false;
			return true;
		});

		net.epochs = 150;

		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());
		Console.writeLine("Final Result:");
		double err = printConfusionMatrix(net, dataSets[2].examples());
		Console.writeLine("Test set accuracy:    " + Format.sprintf("%1.8f", (1 - err)));
		saveErrorImages(net, dataSets[2].examples());

		Console.writeLine("");
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

		DataSet[] dataSets = loadImageDataSets(trainDirectory, tuneDirectory, testDirectory, imageSize,3);

		long end = System.nanoTime() - start;

		Example ex = dataSets[0].get(0);

		ConvNet net = new ConvNet();

		net.addLayer(new Input(ex.x.width(), ex.x.height(), ex.x.depth()));

		net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));
		net.addLayer(new DropOut(0.5));

		net.addLayer(new FullConnect(ex.y.depth(), 1.0));
		net.addLayer(new Softmax());

		double eta = 0.007;
		double alpha = 0.90;
		double lambda = 0.0001;

		Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.005, lambda);

		trainer.onEpoch(t -> {
			Console.writeLine("Epoch: " + t.epoch());
			double err = printConfusionMatrix(t.net(), dataSets[1].examples());
			Console.writeLine("Tune set accuracy: " + Format.sprintf("%1.8f", (1 - err)));
			Console.writeLine("");
			if (err < 0.21) return false;
			return true;
		});

		net.epochs = 150;
		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

		Console.writeLine("Final Result:");
		double err = printConfusionMatrix(net, dataSets[2].examples());
		Console.writeLine("Accuracy:    " + Format.sprintf("%1.8f", (1 - err)));
		saveErrorImages(net, dataSets[2].examples());

		Console.writeLine("");

	}

}
