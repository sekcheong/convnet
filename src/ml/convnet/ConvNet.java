package ml.convnet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.convnet.layer.*;

public class ConvNet {
	public List<Map<String, Object>> _layerSpec = new ArrayList<Map<String, Object>>();


	public Map<String, Object> addLayer() {
		Map<String, Object> params = new HashMap<String, Object>();
		_layerSpec.add(params);
		return params;
	}


	public Layer[] layers() throws Exception {
		List<Layer> layers = new ArrayList<Layer>();
		Layer layer;
		Layer prev;

		String act;
		int w;
		int h;
		int d;
		int units;
		double bias;

		for (int i = 0; i < _layerSpec.size(); i++) {

			Map<String, Object> p = _layerSpec.get(i);

			LayerType layerType = (LayerType) p.get("type");
			if (layerType == null) {
				throw new Exception("layer type not specified for layer " + i);
			}

			switch (layerType) {
				case input:
					w = (int) p.get("width");
					h = (int) p.get("height");
					d = (int) p.get("depth");
					layer = new InputLayer(w, h, d);
					layers.add(layer);
					break;

				case fullconnect:
					prev = layers.get(layers.size() - 1);
					units = (int) p.get("units");
					if (p.containsKey("bias")) {
						bias = (double) p.get("bias");
					}
					else {
						bias = prev.bias;
					}
					//layer = new FullConnectLayer(prev.outW(), prev.outH(), prev.outD(), units, bias);
				//	layers.add(layer);
					act = (String) p.get("activation");
					if (act == null || act.trim().length() == 0) {
						throw new Exception("activation is not specifed for layer " + i);
					}
					layer = getActivationLayer(act, p);
					layers.add(layer);
					break;

				case convolution:

				case pool:

				case regression:

				case softmax:

				case leru:

				case sigmoid:

				case htan:

				case dropout:

				case maxout:

				default:
					break;
			}

		}

		return null;
	}


	private Layer getActivationLayer(String act, Map<String, Object> p) {
		// TODO Auto-generated method stub
		return null;
	}

}
