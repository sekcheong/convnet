package ml.convnet.layers;

import ml.convnet.ConvNet;

public abstract class Layer {
	private ConvNet _net;
	private LayerType _type;
	private int _inWidth;
	private int _inHeight;
	private int _inDepth;
	private int _outWidth;
	private int _outHeight;
	private int _outDepth;


	abstract void forward();


	abstract void backward();


	public void net(ConvNet net) {
		_net = net;
	}


	public ConvNet net() {
		return _net;
	}


	public LayerType type() {
		return _type;
	}


	public void type(LayerType type) {
		_type = type;
	}


	public int outputDepth() {
		return _outDepth;
	}


	public void outputDepth(int v) {
		this._outDepth = v;
	}


	public int outputHeight() {
		return _outHeight;
	}


	public void outputHeight(int v) {
		this._outHeight = v;
	}


	public int outWidth() {
		return _outWidth;
	}


	public void outWidth(int v) {
		this._outWidth = v;
	}


	public int inputDepth() {
		return _inDepth;
	}


	public void inputDepth(int v) {
		this._inDepth = v;
	}


	public int inputHeight() {
		return _inHeight;
	}


	public void inputHeight(int v) {
		this._inHeight = v;
	}


	public int inputWidth() {
		return _inWidth;
	}


	public void inputWidth(int v) {
		_inWidth = v;
	}

}