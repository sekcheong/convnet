package ml.convnet.layer.loss;

import ml.convnet.Cube;
import ml.convnet.layer.Layer;
import ml.convnet.layer.LayerType;

public class SoftmaxLayer extends Layer {
	private int _ncls;
	private double[] _es;


	public SoftmaxLayer(Layer prev, int nclass) {
		super(prev);
		this.inW(prev.outW());
		this.inH(prev.outH());
		this.inD(prev.outD());
		this.outW(1);
		this.outH(1);
		this.outD(this.inLength());
		_ncls = nclass;
		this.type = LayerType.softmax;
	}


	public Cube forward(Cube x) {
		this.output = x;

		Cube A = new Cube(1, 1, this.outD(), 0.0);

		// compute max activation
		double[] as = x.W;
		double amax = x.W[0];
		for (int i = 1; i < this.outD(); i++) {
			if (as[i] > amax) amax = as[i];
		}

		// compute exponentials (carefully to not blow up)
		double[] es = new double[this.outD()];
		double esum = 0.0;
		for (int i = 0; i < es.length; i++) {
			double e = Math.exp(as[i] - amax);
			esum += e;
			es[i] = e;
		}

		// normalize and output to sum to one
		for (int i = 0; i < es.length; i++) {
			es[i] /= esum;
			A.W[i] = es[i];
		}

		_es = es; // save these for backprop
		this.output = A;
		return A;
	}


	public Object backward(Object v) {
		int y = (int) v;
		// compute and accumulate gradient wrt weights and bias of this layer
		Cube x = this.input;
		x.dW = new double[x.W.length]; // zero out the gradient of input Cube

		for (int i = 0; i < this.outD(); i++) {
			double indicator = (i == y) ? 1.0 : 0.0;
			double mul = -(indicator - _es[i]);
			x.dW[i] = mul;
		}
		// loss is the class negative log likelihood
		return -Math.log(_es[y]);
	}

}
