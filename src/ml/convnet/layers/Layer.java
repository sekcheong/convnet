package ml.convnet.layers;

import ml.convnet.ConvNet;

public abstract class Layer {
	private ConvNet _net;
	private LayerType _type;
	
	abstract void forward();
	
	abstract void backward();
	
	public void net(ConvNet net) {
		_net = net;
	}
	
	public ConvNet net() {
		return _net;
	}
	
	public LayerType type() {
		return _type;
	}
	
	public void type(LayerType type) {
		_type = type;
	}

}