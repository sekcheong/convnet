package ml.convnet.layer.loss;

import ml.convnet.Volume;
import ml.convnet.layer.LayerType;

public class Softmax extends LossLayer {

	private double[] _es;
	private int _classes;

	public Softmax() {
		this.type = LayerType.softmax;
	}

	public Softmax(int classes) {
		this.type = LayerType.softmax;
		_classes = classes;
	}

	public Volume forward(Volume V) {
		this.input = V;

		Volume A = new Volume(1, 1, this.outD(), 0.0);

		// compute max activation
		double[] as = V.W;
		double amax = V.W[0];
		for (int i = 1; i < this.outD(); i++) {
			if (as[i] > amax) amax = as[i];
		}

		// compute exponentials (carefully to not blow up)
		double[] es = new double[this.outD()];
		double esum = 0.0;
		for (int i = 0; i < this.outD(); i++) {
			double e = Math.exp(as[i] - amax);
			esum += e;
			es[i] = e;
		}

		// normalize and output to sum to one
		for (int i = 0; i < this.outD(); i++) {
			es[i] /= esum;
			A.W[i] = es[i];
		}

		this._es = es; // save these for backprop
		this.output = A;
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

		// compute the class from vector v;
		Volume x = this.input;
		x.dW = new double[x.W.length];

		for (int i = 0; i < this.outD(); i++) {
			double indicator = (i == y) ? 1.0 : 0.0;
			double mul = -(indicator - this._es[i]);
			x.dW[i] = mul;
		}

		// loss is the class negative log likelihood
		return -Math.log(this._es[y]);
	}

}
