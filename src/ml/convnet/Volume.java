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


	/**
	 * Creates a volume and initializes it with random weights
	 * 
	 * @param width
	 *            The width of the volume
	 * @param height
	 *            The height of the volume
	 * @param depth
	 *            The depth of the volume
	 */
	public Volume(int width, int height, int depth) {
		createVolumeWithRandom(width, height, depth);
	}


	/**
	 * Creates a volume with a specified dimension and fills it will a constant
	 * 
	 * @param width
	 *            The width of the volume
	 * @param height
	 *            The height of the volume
	 * @param depth
	 *            The depth of the volume
	 * @param c
	 *            The constant the volume will be filled with
	 */
	public Volume(int width, int height, int depth, double c) {
		createVolumeWithConst(width, height, depth, c);
	}


	/**
	 * Creates a volume and fills it will a constant
	 *
	 * @param v
	 *            The volume whose width, height, and depth will be used for creating the new volume
	 * @param c
	 *            The constant the volume will be filled with
	 */
	public Volume(Volume v, double c) {
		createVolumeWithConst(v.dim[0], v.dim[1], v.dim[2], c);
	}


	/**
	 * Creates a copy of a given volume
	 * 
	 * @param src
	 *            The source volume
	 */
	public Volume(Volume src) {
		for (int i = 0; i < src.dim.length; i++) {
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


	/**
	 * Creates a volume and populates it with a given vector.
	 * 
	 * @param x
	 *            the array of double to populate the volume's W
	 */
	public Volume(double[] x) {
		dim[0] = 1;
		dim[1] = 1;
		dim[2] = x.length;
		W = new double[x.length];
		dW = new double[x.length];
		for (int i = 0; i < x.length; i++) {
			W[i] = x[i];
		}
	}


	/**
	 * Creates a volume give given dimension and populates it with the content of
	 * a given array
	 * 
	 * @param x
	 *            the array of double to populate the volume
	 */
	public Volume(int width, int height, int depth, double[] x) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		for (int i = 0; i < W.length; i++) {
			W[i] = x[i];
		}
	}


	private void createVolumeWithRandom(int width, int height, int depth) {
		dim[0] = width;
		dim[1] = height;
		dim[2] = depth;
		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		initRandomWeights(W);
	}


	private void createVolumeWithConst(int width, int height, int depth, double c) {
		this.dim[0] = width;
		this.dim[1] = height;
		this.dim[2] = depth;

		W = new double[dim[0] * dim[1] * dim[2]];
		dW = new double[W.length];
		if (c == 0.0) return;
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


	public Volume normalize() {
		double min;
		double max;
		
		Volume v = new Volume(this);
		double[] u = v.W;
		
		min = u[0];
		max = u[1];
		for (int i = 0; i < u.length; i++) {
			if (u[i] > max) max = u[i];
			if (u[i] < min) min = u[i];
		}
		
		double z = (max - min);
		for (int i = 0; i < u.length; i++) {
			u[i] = (u[i] - min) / z;
		}
		return v;
	}

}