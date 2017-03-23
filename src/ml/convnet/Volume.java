package ml.convnet;

import java.util.Random;

public class Volume {

	// the weight parameters
	public double[] W;

	// the gradients
	public double[] dW;

	// the dimension of the volume dim = [w, h, d]
	public int[] dim = new int[3];

	private static Random rand = new Random();


	public Volume(int width, int height, int depth) {
		createVolumeWithRandom(width, height, depth);
	}


	public Volume(int width, int height, int depth, double c) {
		createVolumeWithConst(width, height, depth, c);
	}


	public Volume(Volume v, double c) {
		createVolumeWithConst(v.dim[0], v.dim[1], v.dim[2], c);
	}


	public Volume(Volume src) {
		for (int i = 0; i < src.dim.length; i++) {
			dim[i] = src.dim[i];
		}

		W = new double[src.W.length];
		for (int i = 0; i < src.W.length; i++) {
			W[i] = src.W[i];
		}

		if (src.dW != null) {
			dW = new double[src.dW.length];
			for (int i = 0; i < src.dW.length; i++) {
				dW[i] = src.dW[i];
			}
		}
	}


	public Volume(double[] x) {
		dim[0] = 1;
		dim[1] = 1;
		dim[2] = x.length;
		W = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			W[i] = x[i];
		}
	}


	public Volume(int width, int height, int depth, double[] x) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		for (int i = 0; i < W.length; i++) {
			W[i] = x[i];
		}
	}


	private void createVolumeWithRandom(int width, int height, int depth) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		initRandomWeights(W);
	}


	private void createVolumeWithConst(int width, int height, int depth, double c) {
		this.dim[0] = width;
		this.dim[1] = height;
		this.dim[2] = depth;

		W = new double[dim[0] * dim[1] * dim[2]];
		if (c == 0) return;
		for (int i = 0; i < W.length; i++) {
			W[i] = c;
		}
	}


	private static void initRandomWeights(double[] w) {
		double scale = Math.sqrt(1.0 / ((double) (w.length)));
		for (int i = 0; i < w.length; i++) {
			w[i] = rand.nextGaussian() * scale;
		}
	}


	public int index(int x, int y, int z) {
		return ((dim[0] * y) + x) * dim[2] + z;
	}


	public double get(int x, int y, int z) {
		int i = index(x, y, z);
		return W[i];
	}


	public void set(int x, int y, int z, double v) {
		W[index(x, y, z)] = v;
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


	public void add(Volume v) {
		add(v.W);
	}


	public void add(double[] d) {
		for (int i = 0; i < W.length; i++) {
			W[i] += d[i];
		}
	}


	public double dot(Volume v) {
		return dot(v.W);
	}


	public double dot(double[] v) {
		double y = 0;
		for (int i = 0; i < W.length; i++) {
			y = y + W[i] * v[i];
		}
		return y;
	}


	public void addScale(Volume v, double scale) {
		addScale(v.W, scale);
	}


	public void addScale(double[] d, double scale) {
		for (int i = 0; i < W.length; i++)
			W[i] += d[i] * scale;

	}


	public void addGrad(int x, int y, int z, double grad) {
		dW[index(x, y, z)] += grad;
	}


	public double getGrad(int x, int y, int z) {
		return dW[index(x, y, z)];
	}


	public void setGrad(int x, int y, int z, double grad) {
		dW[index(x, y, z)] = grad;
	}


	public double dotGrad(Volume v) {
		return dotGrad(v.dW);
	}


	public double dotGrad(double[] v) {
		double y = 0;
		for (int i = 0; i < dW.length; i++) {
			y = y + dW[i] * v[i];
		}
		return y;
	}

}