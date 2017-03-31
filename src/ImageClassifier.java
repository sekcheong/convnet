import java.util.ArrayList;
import java.util.List;

import ml.convnet.ConvNet;
import ml.convnet.Volume;
import ml.convnet.layer.*;
import ml.convnet.layer.activation.*;
import ml.convnet.layer.loss.*;
import ml.convnet.trainer.*;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.image.*;
import ml.data.image.ImageUtil.*;
import ml.utils.Console;
import ml.utils.Format;
import ml.utils.tracing.StopWatch;

public class ImageClassifier {

	private static String[] _cats = new String[] { "airplane", "butterfly", "flower", "piano", "starfish", "watch" };


	private static DataSet[] loadImageDataSets(String trainDir, String tuneDir, String testDir, int imageSize, LoadOption option) {
		ImageDataSetReader train = new ImageDataSetReader(trainDir, _cats, imageSize, option);
		ImageDataSetReader tune = new ImageDataSetReader(tuneDir, _cats, imageSize, option);
		ImageDataSetReader test = new ImageDataSetReader(testDir, _cats, imageSize, option);
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
		for (int i = 1; i < y.length; i++) {
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


	private static void learningCurve(DataSet[] dataSets) {

		Example[] train = dataSets[0].examples();
		Example[] tune = dataSets[1].examples();
		Example[] test = dataSets[2].examples();

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
			
			net.epochs = 50;

			trainer.train(net, train, tune);

			Console.writeLine("Tune examples: " + tune.length);
			double err = printConfusionMatrix(net, tune);
			Console.writeLine("Test set accuracy: " + Format.sprintf("%1.8f", (1 - err)));

			Console.writeLine("Tune examples: " + test.length);
			err = printConfusionMatrix(net, test);
			Console.writeLine("Test set accuracy: " + Format.sprintf("%1.8f", (1 - err)));

			Console.writeLine("");

		}
	}


	public static void trainAndTest(DataSet[] dataSets, int epochs) {

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
			double trainerr;
			double testerr;
			double tuneerr;

			Console.writeLine("Train size: " + dataSets[0].examples().length);
			trainerr = printConfusionMatrix(net, dataSets[0].examples());
			Console.writeLine("Train accuracy: " + Format.sprintf("%1.8f", (1 - trainerr)));
			Console.writeLine("");

			Console.writeLine("Tune size: " + dataSets[1].examples().length);
			tuneerr = printConfusionMatrix(net, dataSets[1].examples());
			Console.writeLine("Tune accuracy: " + Format.sprintf("%1.8f", (1 - tuneerr)));
			Console.writeLine("");

			Console.writeLine("Test size: " + dataSets[2].examples().length);
			testerr = printConfusionMatrix(net, dataSets[2].examples());
			Console.writeLine("Test accuracy: " + Format.sprintf("%1.8f", (1 - testerr)));
			Console.writeLine("");
			Console.writeLine("");
			
			Volume[] filters = net.layers()[1].response();
			ImageUtil.saveFilters(filters, 5, "./images/epoch_" + t.epoch() + "_l1_filters" + ".png");
			ImageUtil.saveVolumeLayers(net.layers()[2].output, 5, "./images/epoch_" + t.epoch() + "_l1_activation" + ".png");
									

			//if (trainerr == 0.0) return false;

			return true;
		});


		net.epochs = epochs;
		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

		saveErrorImages(net, dataSets[2].examples());

		Console.writeLine("");
	}


	public static void main(String[] args) {

		String trainDirectory = "./data/images/trainset/";
		String tuneDirectory = "./data/images/tuneset/";
		String testDirectory = "./data/images/testset/";

		int imageSize = 32;

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

		StopWatch timer = new StopWatch();
		timer.start();
		DataSet[] dataSets = loadImageDataSets(trainDirectory, tuneDirectory, testDirectory, imageSize, LoadOption.RGB_EDGES);
		timer.stop();

		Console.writeLine("Data sets loading time: " + timer.elapsedTime() + "sec");

		trainAndTest(dataSets, 500);

	}

}
