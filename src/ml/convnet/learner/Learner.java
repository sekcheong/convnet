package ml.convnet.learner;

import ml.convnet.ConvNet;


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


	public double costLoss() {
		return 0;
	}


	public double decayLossL1() {
		return 0;
	}


	public double decayLossL2() {
		return 0;
	}

	public int forwardTime() {
		return 0;
	}
	
	public int backwardTime() {
		return 0;
	}

	
	public void net(ConvNet convNet) {
		_net = convNet;
	}


	public ConvNet net() {
		return _net;
	}
	
	public abstract void train(double[] x, double[] y);

}
