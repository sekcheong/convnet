package ml.convnet.layer;

import ml.convnet.ConvNet;
import ml.convnet.Cube;

public abstract class Layer {
	public ConvNet net;
	public LayerType type;
	public int index;

	public Cube input;
	public Cube output;
	public Cube biases;

	public void backward() {

	}

	public Cube forward(Cube x) {
		return null;
	}


	public Cube backward(Object v) {
		return null;
	}


	public double backward(double[] v) {
		return 0;
	}


	public double[][][] getResponse() {
		return null;
	}
	
}