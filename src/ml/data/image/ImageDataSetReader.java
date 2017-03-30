package ml.data.image;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import ml.convnet.Volume;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.image.ImageUtil.LoadOption;
import ml.io.DataReader;

public class ImageDataSetReader extends DataReader {

	private String _fileDir;
	private int _size;
	private String[] _cats;
	private LoadOption _option;


	public ImageDataSetReader(String fileDir, String[] categories, int size) {
		init(fileDir, categories, size, LoadOption.RGB);

	}


	public ImageDataSetReader(String fileDir, String[] categories, int size, LoadOption option) {
		init(fileDir, categories, size, option);
	}


	private void init(String fileDir, String[] categories, int size, LoadOption option) {
		_fileDir = fileDir;
		_size = size;
		_cats = categories;
		_option = option;
		for (int i = 0; i < categories.length; i++) {
			categories[i] = categories[i]	.trim()
											.toLowerCase();
		}
	}


	private int getCatNumber(String name) {
		name = name	.trim()
					.toLowerCase();
		for (int i = 0; i < _cats.length; i++) {
			if (_cats[i].compareTo(name) == 0) return i;
		}
		return -1;
	}


	private Volume imageNameToVolume(String name) {

		name = name.toLowerCase();
		int cat = 0;

		for (int i = 0; i < _cats.length; i++) {
			if (name.contains(_cats[i])) {
				cat = i;
				break;
			}
		}

		double[] y = new double[_cats.length];
		y[cat] = 1.0;
		return new Volume(y);
	}


	private Example imageToExample(String name, BufferedImage image, LoadOption options) {
		Volume x = ImageUtil.imageToVolume(image, options);
		x.zeroMean();
		Volume y = imageNameToVolume(name);
		return new Example(x, y);
	}


	@Override
	public DataSet readDataSet() {

		List<Example> examples = new ArrayList<Example>();

		File dir = new File(_fileDir);

		if (_size <= 0) _size = 32;

		for (File file : dir.listFiles()) {

			if (!file.isFile()) continue;

			String fileName = file	.getName()
									.toLowerCase();
			if (!(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".png"))) continue;

			try {
				BufferedImage img = ImageIO.read(file);
				if (img.getWidth() != _size || img.getHeight() != _size) {
					img = ImageUtil.scaleImage(img, _size, _size);
				}
				Example e = imageToExample(fileName, img, _option);
				examples.add(e);
				// saveImageLayer(e.x, 3,  "./bin/images/z" + fileName + ".png");
				// ImageUtil.(e.x, "./bin/images/z" + fileName + "_e.png");
			}
			catch (IOException ex) {
				System.err.println("Error: cannot load in the image file '" + file.getName() + "'");
				System.exit(1);
			}
		}
		Example[] data = examples.toArray(new Example[examples.size()]);
		return new DataSet(data);
	}

}
