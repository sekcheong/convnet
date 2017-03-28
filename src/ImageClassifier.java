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

	private static enum Category {
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
		for (int i = 0; i < y.length; i++) {
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
		int[][] conf = new int[w-1][w-1];
		int err = 0;
		for (Example e : test) {
			double[] yhat = net.predict(e.x.W);
			int p = maxOut(yhat);
			int a = maxOut(e.y.W);
			conf[p][a]++;
			if (p!=a) err++;
		}

		String[] cat = new String[] { "  airplane", 
		                              " butterfly", 
		                              "    flower", 
		                              "     piano", 
		                              "  starfish", 
		                              "     watch" };
		String header = "";
		for (int i=0; i<cat.length; i++) {
			header = header + cat[i];
		}
		Console.writeLine("            airplane  butterfly     flower      piano   starfish      watch");
		
		for (int i = 0; i < conf.length; i++) {
			StringBuffer sb = new StringBuffer();
			for (int j = 0; j < conf[i].length; j++) {
				sb.append(Format.sprintf("%10d", conf[i][j])).append(" ");
			}
			Console.writeLine(cat[i] + sb.toString());
		}
		
		return ((double)err) / test.length;
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

		net.addLayer(new Convolution(5, 5, 25, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 16, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

		net.addLayer(new Convolution(5, 5, 20, 1, 2, 1.0));
		net.addLayer(new LeRu());
		net.addLayer(new Pool(2, 2, 2, 1));

//		net.addLayer(new FullConnect(450, 1.0));
//		net.addLayer(new LeRu());
		net.addLayer(new DropOut(0.5));

		net.addLayer(new FullConnect(ex.y.depth(), 1.0));
		net.addLayer(new Softmax());

		double eta = 0.0005;
		double alpha = 0.90;
		double lambda = 0.00008;

		Trainer trainer = new SGDTrainer(eta, 10, alpha, 0.005, lambda);
		//Trainer trainer = new AdamTrainer(eta, 4, alpha, 0.00005, lambda, 0.9, 0.99, 1e-8);

		trainer.onEpoch(t -> {
			double err = computeError(t.net(), dataSets[1].examples());
			Console.writeLine("epoch: " + t.epoch());
			Console.writeLine("error: " + err);
			return true;
		});

		trainer.onStep(t -> {
			//Console.writeLine("step: " + t.step());
			return true;
		});

		net.epochs = 100;

		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

		double err = printConfusionMatrix(net, dataSets[2].examples());
		Console.writeLine("");
		Console.writeLine("Accuracy: " + (1 - err));
		saveErrorImages(net, dataSets[2].examples());
		
	}

}
