package ml.convnet.layer;

import ml.convnet.Volume;

public class Pool extends Layer {

	private int _w;

	private int _h;

	private int _d;

	private int _stride;

	private int _pad;

	private int[][] _mask;


	public Pool(int w, int h, int stride, int pad) {

		this.type = LayerType.pool;

		int t;
		_w = w;
		_h = (h == 0) ? _w : _h;
		_stride = stride;
		_pad = pad;

	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outD(this.inD());

		int t;
		t = (int) Math.floor((double) (this.inW() + _pad * 2 - _w) / _stride + 1);
		this.outW(t);

		t = (int) Math.floor((double) (this.inH() + _pad * 2 - _h) / _stride + 1);
		this.outH(t);

		// stores mask for x, y coordinates for where the max comes from, for
		// each output neuron
		_mask = new int[this.outLength()][2];

	}


	public Volume forward(Volume v) {
		this.input = v;
		Volume out = new Volume(this.outW(), this.outH(), this.outD(), 0.0);
		this.output = out;

		int n = 0; // a counter for switches
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;

			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;

				for (int ay = 0; ay < this.outH(); y += this._stride, ay++) {
					// convolve centered at this particular location
					double a = -99999999999999.0;
					int winx = -1, winy = -1;
					for (int fx = 0; fx < this._w; fx++) {
						for (int fy = 0; fy < this._h; fy++) {
							int oy = y + fy;
							int ox = x + fx;
							if (oy >= 0 && oy < v.height() && ox >= 0 && ox < v.width()) {
								double q = v.get(ox, oy, d);
								// perform max pooling and store pointers to where
								// the max came from. This will speed up backprop
								// and can help make nice visualizations in future
								if (q > a) {
									a = q;
									winx = ox;
									winy = oy;
								}
							}
						}
					}

					_mask[n][0] = winx;
					_mask[n][1] = winy;
					n++;
					out.set(ax, ay, d, a);
				}
			}
		}

		return out;
	}


	public void backward() {
		Volume in = this.input;

		// pooling layers have no parameters, so simply compute gradient wrt
		// data here
		in.dW = new double[in.W.length];
		Volume A = this.output;

		int n = 0;
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outH(); y += this._stride, ay++) {
					double chainGrad = this.output.getGrad(ax, ay, d);
					in.addGrad(_mask[n][0], _mask[n][1], d, chainGrad);
					n++;
				}
			}

		}
	}

}