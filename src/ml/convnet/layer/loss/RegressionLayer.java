package ml.convnet.layer.loss;

import ml.convnet.Cube;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class RegressionLayer extends Layer {

	private int _inputs;
	private int _outW;
	private int _outH;
	private int _inW;
	private int _inH;
	private int _inD;


	public RegressionLayer(int w, int h, int d) {
		_inW = w;
		_inH = h;
		_inD = d;
		_outW = 1;
		_outH = 1;
		_inputs = _inW * _inH * _inD;
		this.type = LayerType.regression;
	}


	@Override
	public Cube forward(Cube x) {
		this.input = x;
		this.output = x;
		return x;
	}


	public double backward(double[] y) {
		Cube x = this.input;
		double loss = 0;
		for (int i = 0; i < x.W.length; i++) {
			double dy = x.W[i] - y[i];
			x.dW[i] = dy;
			loss += 0.5 * dy * dy;
		}
		return loss;
	}

}
