package ml.convnet.layers;

import ml.convnet.ConvNet;
import ml.convnet.Cube;
import ml.convnet.CubeSize;

public abstract class Layer {
	public ConvNet net;
	public LayerType type;
	public int index;

	public CubeSize inputSize;
	public CubeSize outputSize;

	public Cube input;
	public Cube output;


	public abstract Cube forward(Cube x);


	public abstract Cube backward(Cube y);


	public abstract void backward();

}