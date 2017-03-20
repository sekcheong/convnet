package ml.convnet.layer;

import ml.convnet.Cube;

public class InputLayer extends Layer {

	public InputLayer(int w, int h, int d) {
		super(null);
		this.inW(w).inH(h).inD(d);
		this.outW(w).outH(h).outD(d);
		this.type = LayerType.input;
	}


	public Cube forward(Cube x) {
		this.input = x;
		this.output = x;
		return x;
	}

}