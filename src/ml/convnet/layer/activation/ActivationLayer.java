package ml.convnet.layer.activation;

import ml.convnet.Cube;
import ml.convnet.layer.Layer;

public class ActivationLayer extends Layer {

	public ActivationLayer(Layer prev) {
		super(prev);
		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());
		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());
	}


	public Cube forward(Cube x) {
		this.input = x;
		Cube out = new Cube(x, 0);
		this.output = out;
		double[] f = x.W;
		double[] g = out.W;
		this.computeForward(f, g);
		return out;
	}


	public void backward() {
		Cube in = this.input;
		Cube out = this.output;
		in.dW = new double[in.W.length];
		double[] w = out.W;
		double[] dw = in.dW;
		this.computeBackward(w, dw);
	}


	protected void computeForward(double[] in, double[] out) {
		
	}


	protected void computeBackward(double[] w, double[] dw) {
		
	}

}
