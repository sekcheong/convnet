package ml.convnet.layer;

import ml.convnet.Volume;

public class DropOut extends Layer {

	private boolean[] _dropped;

	private double _dropProb;


	public DropOut(double dropOutProb) {
		this.type = LayerType.dropout;
		_dropProb = dropOutProb;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outW(this.inW());
		this.outH(this.inH());
		this.outD(this.inD());
	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(v);
		int n = v.W.length;

		if (this.training()) {
			this._dropped = new boolean[n];
			for (int i = 0; i < n; i++) {
				if (Math.random() < this._dropProb) {
					out.W[i] = 0;
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
				out.W[i] *= this._dropProb;
			}
		}

		this.output = out;
		return out;
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
