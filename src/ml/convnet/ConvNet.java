package ml.convnet;

import java.util.ArrayList;
import java.util.List;

import ml.convnet.data.Example;
import ml.convnet.layer.*;
import ml.convnet.learner.*;

public class ConvNet  {

	private List<Layer> _layerList = new ArrayList<Layer>();

	private boolean _training;

	private Learner _learner;

	private Layer[] _layers;


	public boolean inTraining() {
		return _training;
	}


	public void addLayer(Layer layer) {
		layer.net(this);
		_layerList.add(layer);
		_layers = null;
	}


	public Layer[] layers() {
		if (_layers == null) {
			_layers = _layerList.toArray(new Layer[_layerList.size()]);
		}
		return _layers;
	}


	public void learner(Learner learner) {
		_learner = learner;
	}


	public Learner learner() {
		return _learner;
	}


	public void train(Example[] train, Example[] tune) {
		
	}


	public void train(Example ex) {
		train(ex.x, ex.y);
	}


	public void train(double[] x, double[] y) {

	}


	public double[] predict(double[] x) {
		return null;
	}


	public double[] accuracy(Example[] test) {
		return null;
	}


	public boolean validate(Example ex) {
		return false;
	}

}