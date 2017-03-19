package ml.convnet;

import java.util.Random;

public class Cube {

	private CubeSize _dim;

	public double[] W;

	public double[] dW;

	private static Random rand = new Random();


	public Cube() {
		_dim = new CubeSize(0, 0, 0);
	}


	public Cube(int width, int height, int depth) {
		initVolume(width, height, depth);
	}


	public Cube(int width, int height, int depth, double v) {
		initVolume(width, height, depth, v);
	}


	public Cube(Cube src) {
		_dim = new CubeSize(src._dim.w, src._dim.h, src._dim.d);
		W = new double[src.W.length];
		for (int i = 0; i < src.W.length; i++) {
			W[i] = src.W[i];
		}

		dW = new double[src.dW.length];
		for (int i = 0; i < src.dW.length; i++) {
			dW[i] = src.dW[i];
		}
	}


	private void initVolume(int width, int height, int depth) {
		_dim = new CubeSize(width, height, depth);
		W = new double[_dim.size()];
		initWeights(W);
	}


	private void initVolume(int width, int height, int depth, double v) {
		_dim = new CubeSize(width, height, depth);
		W = new double[_dim.size()];
		dW = new double[W.length];

		// fill up the weights with the default value
		if (v != 0) {
			for (int i = 0; i < W.length; i++) {
				W[i] = v;
			}
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


	public CubeSize dim() {
		return _dim;
	}


	public double get(int x, int y, int d) {
		int i = _dim.index(x, y, d);
		return this.W[i];
	}


	public void set(int x, int y, int d, double v) {
		int i = _dim.index(x, y, d);
		this.W[i] = v;
	}


	public void set(double c) {
		for (int i = 0; i < W.length; i++)
			W[i] = c;
	}


	public void add(Cube v) {
		for (int i = 0; i < W.length; i++)
			W[i] += v.W[i];
	}


	public void add(double[] d) {
		for (int i = 0; i < W.length; i++)
			W[i] += d[i];
	}


	public void addScale(double[] d, double scale) {
		for (int i = 0; i < W.length; i++)
			W[i] += d[i] * scale;

	}


	public void addScale(Cube v, double scale) {
		for (int i = 0; i < W.length; i++)
			W[i] += v.W[i] * scale;

	}


	public double getGrad(int x, int y, int d) {
		int i = _dim.index(x, y, d);
		return this.dW[i];
	}


	public void setGrad(int x, int y, int d, double grad) {
		int i = _dim.index(x, y, d);
		this.dW[i] = grad;
	}


	public void addGrad(int x, int y, int d, double grad) {
		int i = _dim.index(x, y, d);
		this.dW[i] += grad;
	}


	public static double[] zeros(int size) {
		return new double[size];
	}

}