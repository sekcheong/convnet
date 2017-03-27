import ml.convnet.ConvNet;
import ml.convnet.layer.Convolution;
import ml.convnet.layer.DropOut;
import ml.convnet.layer.FullConnect;
import ml.convnet.layer.Input;
import ml.convnet.layer.Pool;
import ml.convnet.layer.activation.LeRu;
import ml.convnet.layer.activation.Sigmoid;
import ml.convnet.layer.loss.Softmax;
import ml.convnet.trainer.SGDTrainer;
import ml.convnet.trainer.Trainer;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.image.*;
import ml.utils.Console;

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


	public static void main(String[] args) {
		String trainDirectory = "./data/images/trainset/";
		String tuneDirectory = "./data/images/tuneset/";
		String testDirectory = "./data/images/testset/";
		int imageSize = 64;

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

		net.addLayer(new Softmax());

		double eta = 0.0005;
		double epsilon = 0.05;
		double alpha = 0.9;
		double lambda = 0.00008;

		Trainer trainer = new SGDTrainer(eta, 5, alpha, 0.00005, lambda);

		trainer.onEpoch(t -> {			
			return true;
		});

		trainer.onStep(t -> {
			if ((t.step() % 10 == 0)) {

			}
			// Console.writeLine("step: " + t.step());
			return true;
		});

		net.epochs = 1;
		
		trainer.train(net, dataSets[0].examples(), dataSets[1].examples());

	}

}
