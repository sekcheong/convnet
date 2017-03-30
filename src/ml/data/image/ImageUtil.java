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
import ml.utils.Console;

public class ImageUtil {

	public enum LoadOption
	{
		RGB, RGB_EDGES, GRAY, RGB_GRAY, EDGES,
	}


	public static Volume imageToVolume(BufferedImage image) {
		return imageToVolume(image, LoadOption.RGB);
	}


	public static Volume imageToVolume(BufferedImage image, LoadOption option) {
		int width = image.getWidth();
		int height = image.getHeight();
		Volume v = null;

		switch (option) {

		case RGB:
			// RGB volume
			v = new Volume(width, height, 3);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					v.set(i, j, 0, (((double) c.getRed())) / 255);
					v.set(i, j, 1, (((double) c.getGreen())) / 255);
					v.set(i, j, 2, (((double) c.getBlue())) / 255);
				}
			}
			break;

		case GRAY:
			// Gray scale volume
			v = new Volume(width, height, 1);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
					v.set(i, j, 0, g);
				}
			}
			break;

		case RGB_GRAY:
			// RGB and gray scale volume
			v = new Volume(width, height, 4);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					v.set(i, j, 0, (((double) c.getRed())) / 255);
					v.set(i, j, 1, (((double) c.getGreen())) / 255);
					v.set(i, j, 2, (((double) c.getBlue())) / 255);
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
					v.set(i, j, 3, g);
				}
			}
			break;

		case RGB_EDGES:
			// RGB and edges volume
			v = new Volume(width, height, 4);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					v.set(i, j, 0, (((double) c.getRed())) / 255);
					v.set(i, j, 1, (((double) c.getGreen())) / 255);
					v.set(i, j, 2, (((double) c.getBlue())) / 255);
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
					v.set(i, j, 3, g);
				}
			}
			v = sobelFilter(v, 3);
			break;

		case EDGES:
			// Gray scale volume
			v = new Volume(width, height, 1);
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					Color c = new Color(image.getRGB(i, j));
					double g = rgbToGrayScale(c.getRed(), c.getGreen(), c.getBlue());
					v.set(i, j, 0, g);
				}
			}
			v = sobelFilter(v, 0);

			break;
		}
		return v;
	}


	private static double pixelAt(Volume v, int x, int y, int z) {
		if (x < 0 || x >= v.width()) return 0;
		if (y < 0 || y >= v.width()) return 0;
		if (y < 0 || y >= v.width()) return 0;
		if (z < 0 || z >= v.depth()) return 0;
		return v.get(x, y, z);
	}


	private static Volume sobelFilter(Volume v, int z) {
		Volume u = new Volume(v);
		int[][] sobelX = {
				{ -1, 0, 1 },
				{ -2, 0, 2 },
				{ -1, 0, 1 }
		};

		int[][] sobelY = {
				{ -1, -2, -1 },
				{ 0, 0, 0 },
				{ 1, 2, 1 }
		};

		for (int x = 0; x < v.width(); x++) {
			for (int y = 0; y < v.height(); y++) {
				double px = (sobelX[0][0] * pixelAt(v, x - 1, y - 1, z)) + (sobelX[0][1] * pixelAt(v, x, y - 1, z)) + (sobelX[0][2] * pixelAt(v, x + 1, y - 1, z))
						+ (sobelX[1][0] * pixelAt(v, x - 1, y, z)) + (sobelX[1][1] * pixelAt(v, x, y, z)) + (sobelX[1][2] * pixelAt(v, x + 1, y, z))
						+ (sobelX[2][0] * pixelAt(v, x - 1, y + 1, z)) + (sobelX[2][1] * pixelAt(v, x, y + 1, z)) + (sobelX[2][2] * pixelAt(v, x + 1, y + 1, z));

				double py = (sobelY[0][0] * pixelAt(v, x - 1, y - 1, z)) + (sobelY[0][1] * pixelAt(v, x, y - 1, z)) + (sobelY[0][2] * pixelAt(v, x + 1, y - 1, z))
						+ (sobelY[1][0] * pixelAt(v, x - 1, y, z)) + (sobelY[1][1] * pixelAt(v, x, y, z)) + (sobelY[1][2] * pixelAt(v, x + 1, y, z))
						+ (sobelY[2][0] * pixelAt(v, x - 1, y + 1, z)) + (sobelY[2][1] * pixelAt(v, x, y + 1, z)) + (sobelY[2][2] * pixelAt(v, x + 1, y + 1, z));

				double p = Math.sqrt(px * px + py * py);
				u.set(x, y, z, p);
			}
		}
		u = u.normalize(z);
		return u;
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


	private static BufferedImage volumeToImageLayer(Volume v, int layer) {
		BufferedImage image;
		Volume u = v.normalize(layer);
		image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
		for (int i = 0; i < u.height(); i++) {
			for (int j = 0; j < u.width(); j++) {
				int p = (int) (u.get(j, i, layer) * 255);
				p = p + (p << 8) + (p << 16);
				image.setRGB(j, i, p);
			}
		}
		return image;
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
			image = new BufferedImage(u.width(), u.height(), BufferedImage.TYPE_INT_RGB);
			for (int i = 0; i < u.height(); i++) {
				for (int j = 0; j < u.width(); j++) {
					int p = (int) (u.get(j, i, 0) * 255);
					p = p + (p << 8) + (p << 16);
					image.setRGB(j, i, p);
				}
			}
		}
		return image;
	}


	public static Volume distortImage(Volume image) {
		BufferedImage img = volumeToImage(image);
		img = distortImage(img);
		return imageToVolume(img);
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
			e.printStackTrace();
			System.exit(1);
		}
	}


	public static void saveImageLayer(Volume v, int layer, String fileName) {
		BufferedImage image = volumeToImageLayer(v, layer);
		saveImage(image, fileName);
	}


	public static void saveVolumeLayers(Volume v, int cols, String fileName) {
		int pad = 1;
		int rows = v.depth() / cols;
		int r = v.depth() % cols;
		if (r > 0) rows = rows + 1;
		int width = cols * v.width() + pad * (cols + 1);
		int height = rows * v.height() + pad * (rows + 1);
		int ox = pad;
		int oy = pad;

		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		int c = 0xb3b3ff;
		//c = c + (c << 8) + (c << 16);
		for (int i = 0; i < image.getWidth(); i++) {
			for (int j = 0; j < image.getHeight(); j++) {				
				image.setRGB(i, j, c);
			}
		}

		for (int l = 0; l < v.depth(); l++) {
			Volume u = v.normalize(l);
			for (int i = 0; i < u.width(); i++) {
				for (int j = 0; j < u.height(); j++) {
					int p = (int) (u.get(i, j, l) * 255);
					p = p + (p << 8) + (p << 16);
					try {
					image.setRGB(ox + i, oy + j, p);
					}
					catch (Exception ex) {
						Console.writeLine(ex.getMessage());						
					}
				}
			}
			if ((l + 1) % cols == 0) {
				oy = oy + v.height() + pad;
				ox = pad;
			}
			else {
				ox = ox + v.width() + pad;
			}

		}

		image = scaleImage(image, 512, 512);
		saveImage(image, fileName);
	}


	public static Volume loadImage(String fileName, LoadOption options) {
		Volume v = null;
		File inputFile = new File(fileName);
		try {
			BufferedImage image = ImageIO.read(inputFile);
			v = imageToVolume(image, options);
		}
		catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		return v;
	}


	public static BufferedImage imageToBufferedImage(Image img) {
		if (img instanceof BufferedImage) { return (BufferedImage) img; }

		BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bimage.createGraphics();
		g.drawImage(img, 0, 0, null);
		g.dispose();

		return bimage;
	}


	public static BufferedImage scaleImage(BufferedImage img, int width, int height) {
		Image scaledImage = img.getScaledInstance(width, height, java.awt.Image.SCALE_DEFAULT);
		return imageToBufferedImage(scaledImage);
	}

}
