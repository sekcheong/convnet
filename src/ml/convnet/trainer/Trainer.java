package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.convnet.Volume;

public abstract class Trainer {
	protected ConvNet _net;
	protected int _iteration = 0;


	public Trainer() {}


	public int iteration() {
		return _iteration;
	}


	public void incIteration() {
		_iteration++;
	}


	public abstract void train(double[] x, double[] y);


	public void net(ConvNet convNet) {
		_net = convNet;
	}


	public ConvNet net() {
		return _net;
	}

}
