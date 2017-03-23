package ml.convnet;

import java.util.ArrayList;
import java.util.List;

import ml.convnet.data.Example;
import ml.convnet.layer.*;
import ml.convnet.learner.*;

public class ConvNet {

	private List<Layer> _layerList = new ArrayList<Layer>();

	private boolean _training;

	private Learner _learner;

	private Layer[] _layers;


	public ConvNet() {}


	public ConvNet(Learner learner) {
		this.learner(learner);
	}


	public boolean inTraining() {
		return _training;
	}


	public ConvNet addLayer(Layer layer) {
		layer.net(this);
		_layerList.add(layer);
		_layers = null;
		return this;
	}


	public Layer[] layers() {
		if (_layers == null) {
			_layers = _layerList.toArray(new Layer[_layerList.size()]);
		}
		return _layers;
	}


	public void learner(Learner learner) {
		_learner.net(this);
		_learner = learner;
	}


	public Learner learner() {
		return _learner;
	}


	public void train(Example[] train, Example[] tune) {
		_training = true;
	}


	public double[] predict(double[] x) {
		_training = false;
		double[] yhat = forward(x);
		return yhat;
	}


	public double[] accuracy(Example[] test) {
		return null;
	}


	public boolean validate(Example ex) {
		return false;
	}


	public double[] forward(double[] x) {
		Layer[] layers = this.layers();

		Volume act = layers[0].forward(x);
		for (int i = 1; i < layers.length; i++) {
			act = layers[i].forward(act);
		}

		return act.W;
	}


	public double backward(double[] y) {
		Layer[] layers = this.layers();

		double loss = layers[layers.length - 1].backward(y);

		for (int i = layers.length - 2; i >= 0; i--) {
			layers[i].backward();
		}

		return loss;
	}


	public Volume[] response() {
		Layer[] layers = this.layers();
		List<Volume> ret = new ArrayList<Volume>();

		for (int i = 0; i < layers.length; i++) {
			Volume[] layerResponse = layers[i].response();
			for (int j = 0; j < layerResponse.length; j++) {
				ret.add(layerResponse[j]);
			}
		}

		return ret.toArray(new Volume[ret.size()]);
	}

}