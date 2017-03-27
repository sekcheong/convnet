import ml.convnet.ConvNet;
import ml.convnet.layer.*;
import ml.convnet.layer.activation.*;
import ml.convnet.layer.loss.*;
import ml.convnet.trainer.*;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.protein.ProteinReader;
import ml.io.DataReader;
import ml.utils.Console;
import ml.utils.Format;

public class Protein {

	private static void trainXor() {
		ConvNet net = new ConvNet();

		net.addLayer(new Input(1, 1, 2));
		net.addLayer(new FullConnect(3, 0.0));
		net.addLayer(new Sigmoid());
		net.addLayer(new FullConnect(1, 0.0));
		net.addLayer(new Sigmoid());
		net.addLayer(new Softmax());

		Trainer trainer = new SGDTrainer(0.5, 5, 0, 0, 0);

		trainer.onEpoch(t -> {
			//Console.writeLine("epoch: " + t.epoch());
			return true;
		});

		trainer.onStep(t -> {
			//Console.writeLine("step: " + t.step());
			return true;
		});

		Example[] d = new Example[4];

		d[0] = new Example(new double[] { 0, 0 }, new double[] { 0 });
		d[1] = new Example(new double[] { 0, 1 }, new double[] { 1 });
		d[2] = new Example(new double[] { 1, 0 }, new double[] { 1 });
		d[3] = new Example(new double[] { 1, 1 }, new double[] { 1 });

		net.epochs = 25000;
		trainer.train(net, d);
		double[] a = net.predict(new double[] { 0, 1 });
		Console.writeLine(a[0]);

	}


	private static double checkAccuracy(ConvNet net, Example[] test) {
		int correct = 0;
		for (Example t : test) {
			double[] out = net.predict(t.x.W);

			out = threshold(out);

			boolean match = true;
			for (int i = 0; i < t.y.W.length; i++) {
				if (t.y.W[i] != out[i]) {
					match = false;
				}
			}
			if (match) {
				correct++;
			}
		}
		double acc = ((double) correct) / test.length;
		return acc;
	}


	private static void printResults(ConvNet net, Example[] test) {
		int correct = 0;
		Console.writeLine("Actual  Predicted");
		Console.writeLine("=================");
		for (Example t : test) {
			double[] out = net.predict(t.x.W);

			out = threshold(out);

			boolean match = true;
			for (int i = 0; i < t.y.W.length; i++) {
				if (t.y.W[i] != out[i]) {
					match = false;
				}
			}

			if (match) {
				correct++;
			}

			Console.writeLine(getStructureName(t.y.W), "     ", getStructureName(out));
		}

		double acc = ((double) correct) / test.length;
		Console.writeLine("");
		Console.writeLine("Total samples      :", test.length);
		Console.writeLine("Correct prediction :", correct);
		Console.writeLine("Accuracy           :", Format.sprintf("%1.4f", acc * 100), "%");
	}


	private static String getStructureName(double[] v) {
		if (v[0] > 0) return "_";
		if (v[1] > 0) return "h";
		if (v[2] > 0) return "e";
		return "";
	}


	private static double[] threshold(double[] values) {
		double[] t = new double[values.length];
		int max = 0;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > values[max]) {
				max = i;
			}
		}
		t[max] = 1;
		return t;
	}


	private static void predictSecondaryProtein(String dataFile) {

		int hiddenUnits = 9;
		int maxEpoch = 300;

		int repeat = 5;

		DataReader reader = null;

		reader = new ProteinReader(dataFile);
		DataSet data = reader.readDataSet();
		DataSet[] sets = data.split();

		Example[] train = sets[0].examples();
		Example[] tune = sets[1].examples();
		Example[] test = sets[2].examples();

		int inputs = train[0].x.W.length;
		int outputs = train[0].y.W.length;

		ConvNet net = new ConvNet();
		net.addLayer(new Input(1, 1, inputs));
		net.addLayer(new FullConnect(17, 1));
		net.addLayer(new LeRu());
		net.addLayer(new DropOut(0.5));
		net.addLayer(new FullConnect(outputs, 1));
		//net.addLayer(new Sigmoid());
		//net.addLayer(new Regression());
		net.addLayer(new Softmax());

		double eta = 0.005;
		double alpha = 0.9;
		double lambda = 0.00008;

		Trainer trainer = new SGDTrainer(eta, 4, alpha, 0.0, lambda);

		trainer.onEpoch(t -> {
			double acc = checkAccuracy(net, tune);
			if (t.epoch()%5==1) {
			Console.writeLine("epoch: " + t.epoch());
			Console.writeLine("loss: " + t.costLoss());
			Console.writeLine("L1 loss: " + t.decayLossL1());
			Console.writeLine("L2 loss: " + t.decayLossL2());
			Console.writeLine("accuracy: " + acc);
			}
			if (acc>0.64)  {
				return false;
			}
			return true;
		});

		trainer.onStep(t -> {
			if ((t.step() % 10 == 0)) {

			}
			//Console.writeLine("step: " + t.step());
			return true;
		});

		net.epochs = 500;
		trainer.train(net, train, tune);

		printResults(net, test);

	}


	public static void main(String[] args) {
		if (args.length < 1) {
			Console.writeLine("Usage:");
			Console.writeLine("  protein [input]");
			return;
		}
		predictSecondaryProtein(args[0]);
	}
}
