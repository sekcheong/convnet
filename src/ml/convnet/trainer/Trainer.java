package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.data.Example;


public abstract class Trainer {

	protected ConvNet _net;
	protected Example[] _train;
	protected Example[] _tune;

	protected int _iteration = 0;
	protected int _p = 0;
	protected int _epoch = 0;


	public Trainer() {}


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
	

	protected void initExamples(Example[] train) {
		_train = new Example[train.length];
		for (int i = 0; i < train.length; i++) {
			_train[i] = train[i];
		}
	}


	private void shuffle(Example[] train) {
		for (int i = train.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			Example t = train[i];
			train[i] = train[j];
			train[j] = t;
		}
	}


	private Example drawOneExample() {
		Example ex = _train[_p];
		_p++;
		if (_p == _train.length) {
			shuffle(_train);
			_p = 0;
			_epoch++;
		}
		return ex;
	}


	public void train(Example[] train) {
		this.train(train, null);
	}


	public void train(Example[] train, Example[] tune) {
		initExamples(train);
		_tune = tune;
	}


	protected abstract void train(double[] x, double[] y);

}
