package ml.convnet.layer;

import ml.convnet.Volume;

public class Pool extends Layer {

	private int _w;
	private int _h;
	private int _d;
	private int _stride;
	private int _pad;

	private int[] _switchx;
	private int[] _switchy;


	public Pool(int filterW, int filterH, int stride, int pad) {
		this.type = LayerType.pool;
		_w = filterW;
		_h = (filterH == 0) ? _w : filterH;
		_stride = (stride == 0) ? 1 : stride;
		_pad = pad;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outD(this.inD());

		int w = (int) Math.floor((double) (this.inW() + _pad * 2 - _w) / _stride + 1);
		this.outW(w);

		int h = (int) Math.floor((double) (this.inH() + _pad * 2 - _h) / _stride + 1);
		this.outH(h);

		// stores mask for x, y coordinates for where the max comes from, for
		// each output neuron
		_switchx = new int[this.outLength()];
		_switchy = new int[this.outLength()];

	}


	public Volume forward(Volume V) {

		this.input = V;

		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int n = 0; // a counter for switches
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outH(); y += this._stride, ay++) {

					// convolve centered at this particular location
					double a = -99999; // hopefully small enough ;\
					int winx = -1;
					int winy = -1;
					for (int fx = 0; fx < _w; fx++) {
						for (int fy = 0; fy < _h; fy++) {
							int oy = y + fy;
							int ox = x + fx;
							if (oy >= 0 && oy < V.height() && ox >= 0 && ox < V.width()) {
								double v = V.get(ox, oy, d);
								// perform max pooling and store pointers to where
								// the max came from. This will speed up backprop
								// and can help make nice visualizations in future
								if (v > a) {
									a = v;
									winx = ox;
									winy = oy;
								}
							}
						}
					}
					this._switchx[n] = winx;
					this._switchy[n] = winy;
					n++;
					A.set(ax, ay, d, a);
				}
			}
		}

		this.output = A;
		return this.output;
	}


	public void backward() {
		// pooling layers have no parameters, so simply compute
		// gradient wrt data here
		Volume V = this.input;

		// zero out gradient wrt data
		V.dW = new double[V.W.length];
		Volume A = this.output;

		// computed in forward pass
		int n = 0;
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outW(); y += this._stride, ay++) {

					double chain_grad = this.output.getGrad(ax, ay, d);
					V.addGrad(_switchx[n], _switchy[n], d, chain_grad);
					n++;
				}
			}
		}
	}

}