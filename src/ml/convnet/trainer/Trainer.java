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

	boolean _stop = false;


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


	protected Example[] makeTrainExamples(Example[] train) {
		Example[] copy = new Example[train.length];
		for (int i = 0; i < train.length; i++) {
			copy[i] = train[i];
		}
		return copy;
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
		if (_onStep != null) {
			_stop = !_onStep.call(this);
		}
	}


	public void incEpoch() {
		_epoch++;
		if (_onEpoch != null) {
			_stop = !_onEpoch.call(this);
			if (_stop) {
				_stop = true;
			}
		}
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
		_train = makeTrainExamples(train);
		_tune = tune;
		_stop = false;

		while (_epoch < _net.epochs) {
			Example ex = drawOneExample();
			if (_stop) break;
			
			_net.inTraining(true);
			this.trainOneExample(net, ex.x.W, ex.y.W);
			_net.inTraining(false);
			
			this.incStep();
			if (_stop) break;
		}
	}


	protected abstract void trainOneExample(ConvNet net, double[] x, double[] y);

}
