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
		_dropped = new boolean[this.outLength()];
	}


	public Volume forward(Volume V) {
		this.input = V;

		Volume V2 = new Volume(V);
		int N = V.W.length;
		if (this.training()) {

			for (int i = 0; i < N; i++) {
				if (Math.random() < this._dropProb) {
					V2.W[i] = 0;
					this._dropped[i] = true;
				} // drop!
				else {
					this._dropped[i] = false;
				}
			}
		}
		else {
			// scale the activations during prediction
			for (int i = 0; i < N; i++) {
				V2.W[i] *= this._dropProb;
			}
		}

		this.output = V2;
		return output;
	}


	public void backward() {
		Volume V = this.input;
		Volume chain_grad = this.output;
		int N = V.W.length;
		V.dW = new double[N];
		for (int i = 0; i < N; i++) {
			if (!(this._dropped[i])) {
				V.dW[i] = chain_grad.dW[i]; // copy over the gradient
			}
		}
	}

}
