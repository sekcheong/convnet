package ml.convnet.layer;

import ml.convnet.Volume;

public class FullConnect extends Layer {
	private Volume[] _units;
	
	
	public FullConnect(Layer prev, int units, double bias) {
		super(prev);

		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());
		
		this.outH(1);
		this.outW(1);
		this.outD(units);

		_units = new Volume[units];
		for (int i = 0; i < _units.length; i++) {
			_units[i] = new Volume(1, 1, this.inLength());
		}
		this.biases = new Volume(1, 1, this.outD(), bias);
		this.type = LayerType.fullconnect;
	}


	public Volume forward(Volume x) {
		this.input = x;
		Volume out = new Volume(1, 1, this.outD(), 0);
		double[] wx = x.W;
		for (int i = 0; i < this.outD(); i++) {
			double a = 0;
			double[] wi = _units[i].W;
			for (int d = 0; d < this.inLength(); i++) {
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
		Volume in = this.input;
		in.dW = new double[in.W.length];

		// compute gradient wrt weights and data
		for (int i = 0; i < this.outD(); i++) {
			Volume unit_i = _units[i];
			chainGrad = this.output.dW[i];
			for (int d = 0; d < this.inLength(); d++) {
				in.dW[d] += unit_i.W[d] * chainGrad; // grad wrt input data
				unit_i.dW[d] += in.W[d] * chainGrad; // grad wrt params
			}
			this.biases.dW[i] += chainGrad;
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
