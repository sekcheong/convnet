package ml.convnet.layers;

import ml.convnet.Cube;
import ml.convnet.CubeSize;

public class InputLayer extends Layer {
	
	
	public InputLayer(int w, int h, int d) {
		this.inputSize = new CubeSize(w,h,d);
		this.type = LayerType.input;
	}


	@Override
	public Cube forward(Cube x) {
		this.input = x;
		this.output = x;
		return x;
	}


	@Override
	public void backward() {
		
	}


	@Override
	public Cube backward(Cube y) {
		return null;
	}

}
