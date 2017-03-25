package ml.convnet;

import java.util.ArrayList;
import java.util.List;

import ml.convnet.layer.*;
import ml.convnet.trainer.*;
import ml.data.Example;

public class ConvNet {

	private List<Layer> _layerList = new ArrayList<Layer>();

	private boolean _training;

	private Trainer _trainer;

	private Layer[] _layers;

	private Layer _current;
	

	public int epochs;


	public ConvNet() {}


	public boolean inTraining() {
		return _training;
	}


	public ConvNet addLayer(Layer layer) {
		Layer last = null;

		if (_layerList.size() > 0) {
			last = _layerList.get(_layerList.size() - 1);
		}

		layer.net(this);
		_layerList.add(layer);
		
		if (last!=null) {
			last.next(layer);
			layer.last(last);
			layer.connect(last);
		}
		
		//since we modified the layer list we must clear the layer array so it will generate a new one 
		_layers = null;
		return this;
	}


	public Layer[] layers() {
		if (_layers == null) {
			_layers = _layerList.toArray(new Layer[_layerList.size()]);
		}
		return _layers;
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
			Volume[] r = layers[i].response();
			for (int j = 0; j < r.length; j++) {
				ret.add(r[j]);
			}
		}

		return ret.toArray(new Volume[ret.size()]);
	}

}