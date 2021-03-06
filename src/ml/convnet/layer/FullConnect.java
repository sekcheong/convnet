package ml.convnet.layer;

import ml.convnet.Volume;

public class FullConnect extends Layer {

	private Volume[] _units;


	public FullConnect(int units, double bias) {
		this.type = LayerType.fullconnect;
		_units = new Volume[units];
		this.bias = bias;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outH(1);
		this.outW(1);
		this.outD(_units.length);

		for (int i = 0; i < _units.length; i++) {
			_units[i] = new Volume(1, 1, this.inLength());
		}

		this.biases = new Volume(1, 1, this.outD(), this.bias);
	}


	public Volume forward(Volume x) {
		this.input = x;

		Volume out = new Volume(1, 1, this.outD(), 0);
		this.output = out;

		for (int i = 0; i < _units.length; i++) {
			out.W[i] = x.dot(_units[i].W) + this.biases.W[i];
		}

		return out;
	}


	public void backward() {
		double grad;
		Volume in = this.input;
		in.dW = new double[in.W.length];

		for (int i = 0; i < _units.length; i++) {
			Volume unit_i = _units[i];
			grad = this.output.dW[i];
			for (int d = 0; d < this.inLength(); d++) {
				in.dW[d] += unit_i.W[d] * grad;
				unit_i.dW[d] += in.W[d] * grad;
			}
			this.biases.dW[i] += grad;
		}
	}


	public Volume[] response() {
		Volume[] ret = new Volume[_units.length + 1];
		for (int i = 0; i < _units.length; i++) {
			ret[i] = _units[i];
		}
		ret[_units.length] = this.biases;
		return ret;
	}

}
