package ml.convnet.learner;

import ml.convnet.ConvNet;

public abstract class Learner {
	protected ConvNet _net;
	protected int _epochs;

	public Learner() {

	}

	public abstract void train(double[] example, double[] target);

}
