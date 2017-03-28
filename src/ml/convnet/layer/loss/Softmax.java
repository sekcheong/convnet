package ml.convnet.layer.loss;

import ml.convnet.Volume;
import ml.convnet.layer.LayerType;

public class Softmax extends LossLayer {

	private double[] _prob;


	public Softmax() {
		this.type = LayerType.softmax;
	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(1, 1, this.outD(), 0.0);

		double[] as = v.W;
		double amax = v.W[0];

		for (int i = 1; i < this.outD(); i++) {
			if (as[i] > amax) amax = as[i];
		}

		double[] prob = new double[this.outD()];
		double sum = 0.0;
		for (int i = 0; i < this.outD(); i++) {
			double e = Math.exp(as[i] - amax);
			sum += e;
			prob[i] = e;
		}


		for (int i = 0; i < this.outD(); i++) {
			prob[i] /= sum;
			out.W[i] = prob[i];
		}

		this._prob = prob;
		this.output = out;
		return this.output;

	}


	public double backward(double[] v) {
		double max = v[0];
		int y = 0;
		for (int i = 1; i < v.length; i++) {
			if (v[i] > max) {
				y = i;
				max = v[i];
			}
		}

		Volume x = this.input;
		x.dW = new double[x.W.length];

		for (int i = 0; i < this.outD(); i++) {
			double indicator = (i == y) ? 1.0 : 0.0;
			double c = -(indicator - this._prob[i]);
			x.dW[i] = c;
		}

		return -Math.log(this._prob[y]);
	}

}
