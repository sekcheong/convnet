import ml.data.DataSet;
import ml.data.image.*;
import ml.utils.Console;

public class ImageClassifier {



	private static enum Category {
		airplane,
		butterfly,
		flower,
		piano,
		starfish,
		watch
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

	}

}
