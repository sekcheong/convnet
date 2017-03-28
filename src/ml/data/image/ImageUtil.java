package ml.data.image;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import ml.convnet.Volume;

public class ImageUtil {

	public static Volume imageToVolume(BufferedImage image) {
		return imageToVolume(image, false);
	}


	public static Volume imageToVolume(BufferedImage image, boolean toGrayScale) {
		int width = image.getWidth();
		int height = image.getHeight();
		Volume v;

		if (!toGrayScale) {
			v = new Volume(width, height, 3);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					// normalize pixel value to [-0.5, +0.5]
					v.set(i, j, 0, (((double) c.getRed())) / 255);
					v.set(i, j, 1, (((double) c.getGreen())) / 255);
					v.set(i, j, 2, (((double) c.getBlue())) / 255);
				}
			}
		}
		else {
			v = new Volume(width, height, 1);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
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


	public static int rgbToInt(double r, double g, double b) {
		int rgb = (int) (r * 255);
		rgb = (rgb << 8) + (int) (g * 255);
		rgb = (rgb << 8) + (int) (b * 255);
		return rgb;
	}


	public static BufferedImage volumeToImage(Volume v) {
		BufferedImage image;
		Volume u = v.normalize();

		if (u.depth() > 1) {
			image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
			for (int i = 0; i < u.height(); i++) {
				for (int j = 0; j < u.width(); j++) {
					double r = u.get(j, i, 0);
					double g = u.get(j, i, 1);
					double b = u.get(j, i, 2);
					image.setRGB(j, i, rgbToInt(r, g, b));
				}
			}
		}
		else {
			image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_BYTE_GRAY);
			for (int i = 0; i < u.height(); i++) {
				for (int j = 0; j < u.width(); j++) {
					int c = (int) (u.get(j, i, 0)) * 255;
					image.setRGB(j, i, c);
				}
			}
		}
		return image;
	}


	public static BufferedImage distortImage(BufferedImage image) {

		int r = (int) (Math.random() * 2);
		switch (r) {
			case 0:
				// Flip the image vertically
				AffineTransform tx = AffineTransform.getScaleInstance(1, -1);
				tx.translate(0, -image.getHeight(null));
				AffineTransformOp op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;

			case 1:
				// Flip the image horizontally
				tx = AffineTransform.getScaleInstance(-1, 1);
				tx.translate(-image.getWidth(null), 0);
				op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;
			case 2:
				// Flip the image vertically and horizontally; equivalent to rotating the image 180 degrees
				tx = AffineTransform.getScaleInstance(-1, -1);
				tx.translate(-image.getWidth(null), -image.getHeight(null));
				op = new AffineTransformOp(tx, AffineTransformOp.TYPE_NEAREST_NEIGHBOR);
				image = op.filter(image, null);
				break;
		}

		return image;
	}


	public static Volume distortImage(Volume image) {
		BufferedImage img = volumeToImage(image);
		img = distortImage(img);
		return imageToVolume(img);
	}


	public static void saveImage(Volume v, String fileName) {
		BufferedImage image = volumeToImage(v);
		saveImage(image, fileName);
	}


	public static void saveImage(BufferedImage image, String fileName) {
		File outputfile = new File(fileName);
		try {
			ImageIO.write(image, "png", outputfile);
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	public static Volume loadImage(String fileName) {
		Volume v = null;
		File inputFile = new File(fileName);
		try {
			BufferedImage image = ImageIO.read(inputFile);
			v = imageToVolume(image);
		}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return v;
	}


	public static BufferedImage imageToBufferedImage(Image img) {
		if (img instanceof BufferedImage) {
			return (BufferedImage) img;
		}

		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_RGB);

		Graphics2D g = bimage.createGraphics();
		g.drawImage(img, 0, 0, null);
		g.dispose();

		// Return the buffered image
		return bimage;
	}


	public static BufferedImage scaleImage(BufferedImage img, int width, int height) {
		Image scaledImage = img.getScaledInstance(width, height, java.awt.Image.SCALE_DEFAULT);
		return imageToBufferedImage(scaledImage);
	}


}
