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
import ml.io.DataReader;

public class ImageDataSetReader extends DataReader {


	private static enum ImageCategory {
		airplane(0),
		butterfly(1),
		flower(2),
		piano(3),
		starfish(4),
		watch(5),
		unknown(6);

		private final int value;


		private ImageCategory(int value) {
			this.value = value;
		}


		public int getValue() {
			return value;
		}
	}


	private String _fileDir;
	private int _size;


	public ImageDataSetReader(String fileDir, int size) {
		_fileDir = fileDir;
		_size = size;
	}


	private Volume imageNameToVolume(String name) {
		ImageCategory cat = ImageCategory.unknown;
		for (ImageCategory c : ImageCategory.values()) {
			String catName = c.toString().toLowerCase();
			if (name.contains(catName)) {
				cat = c;
				break;
			}
		}
		double[] y = new double[ImageCategory.values().length];
		y[cat.value] = 1.0;
		return new Volume(y);
	}


	private Example imageToExample(String name, BufferedImage image) {
		Volume x = ImageUtil.imageToVolume(image);
		Volume y = imageNameToVolume(name);
		return new Example(x, y);
	}


	@Override
	public DataSet readDataSet() {

		List<Example> examples = new ArrayList<Example>();

		File dir = new File(_fileDir);

		if (_size <= 0) _size = 128;

		for (File file : dir.listFiles()) {

			if (!file.isFile()) continue;

			String fileName = file.getName().toLowerCase();
			if (!(fileName.endsWith(".jpg") || fileName.endsWith(".jpeg") || fileName.endsWith(".png"))) continue;

			try {
				BufferedImage img = ImageIO.read(file);
				if (img.getWidth() != _size || img.getHeight() != _size) {
					img = ImageUtil.scaleImage(img, _size, _size);
				}
				Example e = imageToExample(fileName, img);
				examples.add(e);
				BufferedImage im =  ImageUtil.volumeToImage(e.x);
				ImageUtil.saveImage(e.x, "./bin/z_" + fileName + ".png");
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
