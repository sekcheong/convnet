package ml.convnet.layer.activation;

import ml.convnet.Volume;
import ml.convnet.layer.LayerType;

public class Sigmoid extends ActivationLayer {

	public Sigmoid() {
		this.type = LayerType.sigmoid;
	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(v, 0);
		int n = v.W.length;
		double[] outW = out.W;
		double[] inW = v.W;
		for (int i = 0; i < n; i++) {
			outW[i] = 1.0 / (1.0 + Math.exp(-inW[i]));
		}
		this.output = out;
		return out;
	}


	public void backward() {
		Volume in = this.input;
		Volume out = this.output;
		int n = in.W.length;
		in.dW = new double[n];
		for (int i = 0; i < n; i++) {
			double wi = out.W[i];
			in.dW[i] = wi * (1.0 - wi) * out.dW[i];
		}
	}

}
