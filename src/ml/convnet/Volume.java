package ml.convnet;

import java.util.Random;

public class Volume {

	private int _width;

	private int _height;

	private int _depth;

	private double[] _w;

	private double[] _dw;

	private static Random rand = new Random();


	public Volume() {
		_width = 0;
		_height = 0;
		_depth = 0;
	}


	public Volume(int width, int height, int depth) {
		initVolume(width, height, depth);
	}


	public Volume(Volume src) {
		_width = src._width;
		_height = src._height;
		_depth = src._depth;
		_w = new double[src._w.length];
		for (int i = 0; i < src._w.length; i++) {
			_w[i] = src._w[i];
		}
	}


	private void initVolume(int width, int height, int depth) {
		_width = width;
		_height = height;
		_depth = depth;
		_w = new double[_width * _height * _depth];
		_dw = new double[_w.length];
		// initWeights(_w);
	}


	private static void initWeights(double[] w) {
		double scale = Math.sqrt(1.0 / ((double) (w.length)));
		for (int i = 0; i < w.length; i++) {
			w[i] = rand.nextGaussian() * scale;
		}
	}


	public int width() {
		return _width;
	}


	public int height() {
		return _height;
	}


	public int depth() {
		return _depth;
	}


	public void initWeights() {
		initWeights(this._w);
	}


	private int index(int x, int y, int d) {
		return ((_width * _height * y) + x) * _depth + d;
	}


	public double get(int x, int y, int d) {
		int i = index(x, y, d);
		return this._w[i];
	}


	public void set(int x, int y, int d, double v) {
		int i = index(x, y, d);
		this._w[i] = v;
	}


	public void set(double c) {
		for (int i = 0; i < _w.length; i++) {
			_w[i] = c;
		}
	}


	public void add(Volume v) {
		for (int i = 0; i < _w.length; i++) {
			_w[i] += v._w[i];
		}
	}


	public void add(double[] d) {
		for (int i = 0; i < _w.length; i++) {
			_w[i] += d[i];
		}
	}


	public void addScale(double[] d, double scale) {
		for (int i = 0; i < _w.length; i++) {
			_w[i] += d[i] * scale;
		}
	}


	public void addScale(Volume v, double scale) {
		for (int i = 0; i < _w.length; i++) {
			_w[i] += v._w[i] * scale;
		}
	}


	public double getGrad(int x, int y, int d) {
		int i = index(x, y, d);
		return this._dw[i];
	}


	public void setGrad(int x, int y, int d, double v) {
		int i = index(x, y, d);
		this._dw[i] = v;
	}


	public void addGrad(int x, int y, int d, double v) {
		int i = index(x, y, d);
		this._dw[i] += v;
	}


	public Volume clone() {
		return new Volume(this);
	}
}