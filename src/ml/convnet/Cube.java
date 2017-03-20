package ml.convnet;

import java.util.Random;

public class Cube {

	public double[] W;

	public double[] dW;

	public int[] dim = new int[3];

	private static Random rand = new Random();


	public Cube() {}


	public Cube(int width, int height, int depth) {
		createVolume(width, height, depth);
	}


	public Cube(int width, int height, int depth, double v) {
		createVolume(width, height, depth, v);
	}


	public Cube(Cube src) {
		for (int i=0; i<src.dim.length; i++) {
			dim[i] = src.dim[i];
		}

		W = new double[src.W.length];
		for (int i = 0; i < src.W.length; i++) {
			W[i] = src.W[i];
		}

		dW = new double[src.dW.length];
		for (int i = 0; i < src.dW.length; i++) {
			dW[i] = src.dW[i];
		}
	}


	public Cube(Cube v, double c) {
		createVolume(v.dim[0], v.dim[1], v.dim[2], c);
	}


	private void createVolume(int width, int height, int depth) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		initWeights(W);
	}


	private void createVolume(int width, int height, int depth, double v) {
		this.dim[0] = width;
		this.dim[1] = height;
		this.dim[2] = depth;

		W = new double[dim[0] * dim[1] * dim[2]];
		if (v == 0) return;
		for (int i = 0; i < W.length; i++) {
			W[i] = v;
		}
	}


	private static void initWeights(double[] w) {
		double scale = Math.sqrt(1.0 / ((double) (w.length)));
		for (int i = 0; i < w.length; i++) {
			w[i] = rand.nextGaussian() * scale;
		}
	}


	public void initWeights() {
		initWeights(this.W);
	}


	public static int index(int x, int y, int z) {
		return 0;
	}


	public double get(int x, int y, int z) {
		int i = index(x, y, z);
		return this.W[i];
	}


	public void set(int x, int y, int z, double v) {
		this.W[index(x, y, z)] = v;
	}


	public void setAll(double c) {
		for (int i = 0; i < W.length; i++) {
			W[i] = c;
		}
	}
	
	public int width() {
		return dim[0];
	}

	public int height() {
		return dim[1];
	}
	
	public int depth() {
		return dim[2];
	}
	// public void add(Cube v) {
	// for (int i = 0; i < W.length; i++)
	// W[i] += v.W[i];
	// }
	//
	//
	// public void add(double[] d) {
	// for (int i = 0; i < W.length; i++)
	// W[i] += d[i];
	// }
	//
	//
	// public void addScale(double[] d, double scale) {
	// for (int i = 0; i < W.length; i++)
	// W[i] += d[i] * scale;
	//
	// }
	//
	//
	// public void addScale(Cube v, double scale) {
	// for (int i = 0; i < W.length; i++)
	// W[i] += v.W[i] * scale;
	//
	// }
	//
	//
	// public double getGrad(int x, int y, int d) {
	// int i = _dim.index(x, y, d);
	// return this.dW[i];
	// }
	//
	//
	// public void setGrad(int x, int y, int d, double grad) {
	// int i = _dim.index(x, y, d);
	// this.dW[i] = grad;
	// }
	//
	//
	// public void addGrad(int x, int y, int d, double grad) {
	// int i = _dim.index(x, y, d);
	// this.dW[i] += grad;
	// }
	//
	//
	// public static double[] zeros(int size) {
	// return new double[size];
	// }

}