package ml.convnet.layer;

import ml.convnet.Cube;

public class InputLayer extends Layer {
	int _w;
	int _h;
	int _d;


	public InputLayer(int w, int h, int d) {
		_w = w;
		_h = h;
		_d = d;
		this.type = LayerType.input;
	}


	public Cube forward(Cube x) {
		this.input = x;
		this.output = x;
		return x;
	}

}