package ml.convnet;

import ml.convnet.layers.LayerType;

import java.util.ArrayList;
import java.util.List;

import ml.convnet.layers.Layer;

public class ConvNet {
	public List<Layer> _layers = new ArrayList<Layer>();


	public void addLayer(Layer layer) {
		if (_layers.size() == 0 && layer.type != LayerType.input) {
			throw new IllegalArgumentException("The first layer must be an input layer.");
		}
		layer.index = _layers.size();
		layer.net = this;
		_layers.add(layer);
	}


	public List<Layer> layers() {
		return _layers;
	}
}
