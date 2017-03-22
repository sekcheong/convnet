package ml.convnet.layer.activation;

import ml.convnet.Volume;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class Tanh extends Layer {

	public Tanh(Layer prev) {
		super(prev);
		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());

		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());

		this.type = LayerType.tanh;
	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(v, 0);
		int n = v.W.length;
		double[] outW = out.W;
		double[] inW = v.W;
		for (int i = 0; i < n; i++) {
			outW[i] = Math.tanh(inW[i]);
		}
		this.output = out;
		return out;
	}


	public void backward() {
		Volume in = this.input; // we need to set dw of this
		Volume out = this.output;
		int n = in.W.length;
		in.dW = new double[n];
		for (int i = 0; i < n; i++) {
			double outW = out.W[i];
			in.dW[i] = (1.0 - outW * outW) * out.dW[i];
		}
	}

}
