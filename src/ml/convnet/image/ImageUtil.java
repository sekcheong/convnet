package ml.convnet.image;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import ml.convnet.Volume;

public class ImageUtil {

	public static Volume imageToCube(BufferedImage image) {
		return imageToCube(image, false);
	}

	public static Volume imageToCube(BufferedImage image, boolean toGrayScale) {
		int width = image.getWidth();
		int height = image.getHeight();
		Volume v;

		if (!toGrayScale) {
			v = new Volume(width, height, 3);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					// normalize pixel value to [-0.5, +0.5]
					v.set(i, j, 0, (((double) c.getRed()) / 255.0) - 0.5);
					v.set(i, j, 1, (((double) c.getGreen())) / 255.0 - 0.5);
					v.set(i, j, 2, (((double) c.getBlue())) / 255.0 - 0.5);
				}
			}
		}
		else {
			v = new Volume(width, height, 1);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue()) - 0.5;
					v.set(i, j, 0, g);
				}
			}
		}
		return v;
	}


	public static double rgbToGrayScale(int r, int g, int b) {
		double r0 = (double) r / (255);
		double g0 = (double) g / (255);
		double b0 = (double) b / (255);
		double y = .2126 * r0 + .7152 * g0 + .0722 * b0;
		return y;
	}
}
