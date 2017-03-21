package ml.convnet.layer;

import ml.convnet.Cube;

public class FullConnect extends Layer {
	private Cube[] _units;
	

	public FullConnect(Layer prev, int units, double bias) {
		super(prev);
		
		this.inW(prev.outW()).inH(prev.outH()).inD(prev.outD());
		this.outH(1);
		this.outW(1);
		this.outD(units);

		_units = new Cube[units];
		for (int i = 0; i < _units.length; i++) {
			_units[i] = new Cube(1, 1, this.inLength());
		}
		this.biases = new Cube(1, 1, this.outD(), bias);
		this.type = LayerType.fullconnect;
	}


	public Cube forward(Cube x) {
		this.input = x;
		Cube out = new Cube(1, 1, this.outD(), 0);
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
		Cube in = this.input;
		in.dW = new double[in.W.length];

		// compute gradient wrt weights and data
		for (int i = 0; i < this.outD(); i++) {
			Cube unit_i = _units[i];
			chainGrad = this.output.dW[i];
			for (int d = 0; d < this.inLength(); d++) {
				in.dW[d] += unit_i.W[d] * chainGrad; // grad wrt input data
				unit_i.dW[d] += in.W[d] * chainGrad; // grad wrt params
			}
			this.biases.dW[i] += chainGrad;
		}
	}


	public double[][][] response() {
		int n = this.outD();
		double[][][] res = new double[n + 1][2][];
		for (int i = 0; i < n; i++) {
			res[i][0] = _units[i].W;
			res[i][1] = _units[i].dW;
		}
		res[n][0] = this.biases.W;
		res[n][1] = this.biases.dW;
		return res;
	}

}
