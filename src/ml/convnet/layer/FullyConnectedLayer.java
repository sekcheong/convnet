package ml.convnet.layer;

import ml.convnet.Cube;

public class FullyConnectedLayer extends Layer {
	private Cube[] _filters;
	private int _inputs;
	int _outW;
	int _outH;
	int _outD;


	public FullyConnectedLayer(int w, int h, int d, int units, double bias) {

		_outW = 1;
		_outH = 1;
		_outD = units;
		_inputs = w * h * d;

		_filters = new Cube[_outD];
		for (int i = 0; i < _filters.length; i++) {
			_filters[i] = new Cube(1, 1, _inputs);
		}
		this.biases = new Cube(1, 1, _outD, bias);
		this.type = LayerType.fullyconnected;
	}


	public Cube forward(Cube x) {
		this.input = x;
		Cube out = new Cube(1, 1, _outD, 0);
		double[] wx = x.W;
		for (int i = 0; i < _outD; i++) {
			double a = 0;
			double[] wi = _filters[i].W;
			for (int d = 0; d < _inputs; i++) {
				a += wx[d] * wi[d];
			}
			a += this.biases.W[i];
			out.W[i] = a;
		}
		this.output = out;
		return out;
	}


	public void backward() {
		double chainGrad;
		Cube in = this.input;
		in.dW = new double[in.W.length];

		// compute gradient wrt weights and data
		for (int i = 0; i < _outD; i++) {
			Cube tfi = _filters[i];
			chainGrad = this.output.dW[i];
			for (int d = 0; d < this._inputs; d++) {
				in.dW[d] += tfi.W[d] * chainGrad; // grad wrt input data
				tfi.dW[d] += in.W[d] * chainGrad; // grad wrt params
			}
			this.biases.dW[i] += chainGrad;
		}
	}


	public double[][][] getResponse() {
		double[][][] res = new double[_outD + 1][2][];
		int n = _outD;
		for (int i = 0; i < n; i++) {
			res[i][0] = _filters[i].W;
			res[i][1] = _filters[i].dW;
		}
		res[n][0] = this.biases.W;
		res[n][1] = this.biases.dW;
		return res;
	}

}
