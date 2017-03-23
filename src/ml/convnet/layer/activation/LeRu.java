package ml.convnet.layer.activation;

import ml.convnet.Volume;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class LeRu extends Layer {

	public LeRu(Layer prev) {
		super(prev);
		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());

		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());

		this.type = LayerType.leru;
	}


	public Volume forward(Volume x) {
		this.input = x;
		Volume out = new Volume(x);
		int n = x.W.length;
		double[] outW = out.W;
		for (int i = 0; i < n; i++) {
			if (outW[i] < 0) outW[i] = 0;
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
			if (out.W[i] <= 0) in.dW[i] = 0;
			else in.dW[i] = out.dW[i];
		}
	}

}
