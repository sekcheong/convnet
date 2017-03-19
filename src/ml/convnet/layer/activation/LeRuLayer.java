package ml.convnet.layer.activation;

import ml.convnet.Cube;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class LeRuLayer extends Layer {

	public LeRuLayer() {
		this.type = LayerType.leru;
	}


	public Cube forward(Cube x) {
		this.input = x;
		Cube out = new Cube(x);
		int n = x.W.length;
		double[] V2w = out.W;
		for (int i = 0; i < n; i++) {
			if (V2w[i] < 0) V2w[i] = 0;
		}
		this.output = out;
		return out;
	}


	public void backward() {
		Cube in = this.input;
		Cube out = this.output;
		int n = in.W.length;
		in.dW = new double[n];
		for (int i = 0; i < n; i++) {
			if (out.W[i] <= 0) in.dW[i] = 0;
			else in.dW[i] = out.dW[i];
		}
	}

}
