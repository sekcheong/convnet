package ml.convnet.learner;

import ml.convnet.ConvNet;
import ml.convnet.Volume;

public abstract class Learner {
	protected ConvNet _net;
	protected int _iteration = 0;


	public Learner() {}


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
