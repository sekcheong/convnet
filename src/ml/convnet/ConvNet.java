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
		for (int i = 0; i < _layerSpec.size(); i++) {

			Map<String, Object> p = _layerSpec.get(i);

			LayerType layerType = (LayerType) p.get("type");
			if (layerType == null) {
				throw new Exception("layer type not specified for layer " + i);
			}

			switch (layerType) {
				case input:
					layer = new InputLayer((int) p.get("width"), (int) p.get("height"), (int) p.get("depth"));
					layers.add(layer);
					break;

				case fullyconnected:
					layer = new FullyConnectedLayer((int) p.get("width"), (int) p.get("height"), (int) p.get("depth"), (int) p.get("units"), (double) p.get("bias"));
					layers.add(layer);

					String act = (String) p.get("activation");
					if (act == null || act.trim().length() == 0) {
						throw new Exception("activation is not specifed for layer " + i);
					}

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

}
