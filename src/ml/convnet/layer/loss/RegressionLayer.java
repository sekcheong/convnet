package ml.convnet.layer.loss;

import ml.convnet.Cube;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class RegressionLayer extends Layer {

	RegressionLayer(Layer prev) {
		super(prev);
		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());
		
		this.outW(1);
		this.outH(1);
		this.outD(1);
		this.outD(this.inLength());
		
		this.type = LayerType.regression;
	}


	public Cube forward(Cube x) {
		this.input = x;
		this.output = x;
		return x;
	}


	public double backward(double[] y) {
		Cube x = this.input;
		double loss = 0.0;
		for (int i = 0; i < x.W.length; i++) {
			double dy = x.W[i] - y[i];
			x.dW[i] = dy;
			loss += 0.5 * dy * dy;
		}
		return loss;
	}


	public double backward(double y) {
		Cube x = this.input;
		double dy = x.W[0] - y;
		x.dW[0] = dy;
		return .5 * dy * dy;
	}
	
}
