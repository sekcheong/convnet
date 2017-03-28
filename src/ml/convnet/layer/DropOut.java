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


	public Volume forward(Volume x) {
		this.input = x;

		Volume out = new Volume(x);
		int m = x.W.length;

		if (this.training()) {
			for (int i = 0; i < m; i++) {
				if (Math.random() < this._dropProb) {
					out.W[i] = 0;
					this._dropped[i] = true;
				} // drop!
				else {
					this._dropped[i] = false;
				}
			}
		}
		else {
			for (int i = 0; i < m; i++) {
				out.W[i] *= this._dropProb;
			}
		}

		this.output = out;
		return output;
	}


	public void backward() {
		Volume in = this.input;
		Volume grad = this.output;
		
		in.dW = new double[in.W.length];
		for (int i = 0; i < in.W.length; i++) {
			if (!(this._dropped[i])) {
				in.dW[i] = grad.dW[i];
			}
		}
	}

}
