package ml.convnet.layer;

import ml.convnet.Volume;

public class DropOut extends Layer {

	private boolean[] _dropped;
	private double _dropProb;


	public DropOut(Layer prev, double dropOutProb) {
		super(prev);
		this.inW(prev.outW())
				.inH(prev.outH())
				.inD(prev.outD());
		this.outW(this.inW())
				.outH(this.inH())
				.outD(this.inD());
		this.type = LayerType.dropout;
	}


	public Volume forward(Volume V) {
		this.input = V;
		Volume V2 = new Volume(V);
		int n = V.W.length;

		if (this.training()) {

			this._dropped = new boolean[n];
			for (int i = 0; i < n; i++) {
				if (Math.random() < this._dropProb) {
					V2.W[i] = 0;
					this._dropped[i] = true;
				}
				else {
					this._dropped[i] = false;
				}
			}

		}
		else {

			for (int i = 0; i < n; i++) {
				// scale the activations during prediction
				V2.W[i] *= this._dropProb;
			}

		}

		this.output = V2;
		return V2;
	}


	public void backword() {
		Volume V = this.input; // we need to set dw of this
		double[] chainGrad = this.output.dW;
		int n = V.W.length;
		V.dW = new double[n]; // zero out gradient wrt data
		for (int i = 0; i < n; i++) {
			if (!(this._dropped[i])) {
				V.dW[i] = chainGrad[i]; // copy over the gradient
			}
		}
	}
}
