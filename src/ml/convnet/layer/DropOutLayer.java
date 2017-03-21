package ml.convnet.layer;

import ml.convnet.Cube;

public class DropOutLayer extends Layer {

	private boolean[] _dropped;
	private double _dropProb;


	public DropOutLayer(Layer prev, double dropOutProb) {
		super(prev);
		this.inW(prev.outW()).inH(prev.outH()).inD(prev.outD());
		this.outW(this.inW()).outH(this.inH()).outD(this.inD());
		this.type = LayerType.dropout;
	}


	public Cube forward(Cube V) {
		this.input = V;
		Cube V2 = new Cube(V);
		int N = V.W.length;
		if (this.training()) {
			// do dropout
			this._dropped = new boolean[N];
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
		return V2;
	}


	public void backword() {
		Cube V = this.input; // we need to set dw of this
		Cube chainGrad = this.output;
		int N = V.W.length;
		V.dW = new double[N]; // zero out gradient wrt data
		for (int i = 0; i < N; i++) {
			if (!(this._dropped[i])) {
				V.dW[i] = chainGrad.dW[i]; // copy over the gradient
			}
		}
	}
}
