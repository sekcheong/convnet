package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.data.Example;

public abstract class Trainer {

	public interface TrainerEvent {

		void call(Trainer trainer);
	}

	protected ConvNet _net;

	protected Example[] _train;

	protected Example[] _tune;

	protected int _iteration = 0;

	protected int _p = 0;

	protected int _epoch = 0;

	protected TrainerEvent _onEpoch;

	protected TrainerEvent _onStep;


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
			_onEpoch.call(this);
		}
		return ex;
	}


	public void onEpoch(TrainerEvent callback) {
		_onEpoch = callback;
	}


	public void onStep(TrainerEvent callback) {
		_onStep = callback;
	}


	public void train(ConvNet net, Example[] train) {
		this.train(net, train, null);
	}


	public void train(ConvNet net, Example[] train, Example[] tune) {
		_net = net;
		initExamples(train);
		_tune = tune;
		while (_epoch < _net.epochs) {
			Example ex = drawOneExample();
			this.incIteration();
			this.trainOneExample(net, ex.x.W, ex.y.W);
			if (_onStep != null) _onStep.call(this);
		}
	}


	public int epoch() {
		return _epoch;
	}

	
	protected abstract void trainOneExample(ConvNet net, double[] x, double[] y);


}
