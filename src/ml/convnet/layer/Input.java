package ml.convnet.layer;

import ml.convnet.Volume;

public class Input extends Layer {

	public Input(int w, int h, int d) {
		super(null);

		this.inW(w);
		this.inH(h);
		this.inD(d);

		this.outW(w);
		this.outH(h);
		this.outD(d);

		this.type = LayerType.input;
	}


	public Volume forward(double[] x) {
		Volume v = new Volume(this.inW(), this.inH(), this.inW(), x);
		return this.forward(v);
	}


	public Volume forward(Volume x) {
		this.input = x;
		this.output = x;
		return x;
	}

}