package ml.convnet.layer.loss;

import ml.convnet.Volume;
import ml.convnet.layer.LayerType;

public class Regression extends LossLayer {

	Regression() {
		this.type = LayerType.regression;
	}

	public Volume forward(Volume x) {
		this.input = x;
		this.output = x;
		return x;
	}


	public double backward(double[] y) {
		Volume x = this.input;
		double loss = 0.0;
		for (int i = 0; i < x.W.length; i++) {
			double dy = x.W[i] - y[i];
			x.dW[i] = dy;
			loss += 0.5 * dy * dy;
		}
		return loss;
	}


	public double backward(double y) {
		Volume x = this.input;
		double dy = x.W[0] - y;
		x.dW[0] = dy;
		return .5 * dy * dy;
	}
	
}
