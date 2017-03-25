package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.data.Example;

public abstract class Trainer {

	public interface TrainerEvent {

		boolean call(Trainer trainer);
	}

	protected ConvNet _net;

	protected Example[] _train;

	protected Example[] _tune;

	// the n th example
	protected int _n = 0;

	protected int _epoch = 0;

	protected int _step = 0;

	protected TrainerEvent _onEpoch;

	protected TrainerEvent _onStep;


	public Trainer() {}


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
		Example ex = _train[_n];
		_n++;
		if (_n == _train.length) {
			this.shuffle(_train);
			this.incEpoch();
			_n = 0;
		}
		return ex;
	}


	public int step() {
		return _step;
	}


	public void incStep() {
		_step++;
		if (_onStep != null) _onStep.call(this);
	}


	public void incEpoch() {
		_epoch++;
		if (_onEpoch != null) _onEpoch.call(this);
	}


	public void onEpoch(TrainerEvent callback) {
		_onEpoch = callback;
	}


	public void onStep(TrainerEvent callback) {
		_onStep = callback;
	}


	public int epoch() {
		return _epoch;
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
			this.trainOneExample(net, ex.x.W, ex.y.W);
			this.incStep();
		}
	}


	protected abstract void trainOneExample(ConvNet net, double[] x, double[] y);

}
