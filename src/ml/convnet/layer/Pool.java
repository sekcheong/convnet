package ml.convnet.layer;

import ml.convnet.Volume;
import ml.utils.Console;

public class Pool extends Layer {

	private int _w;
	private int _h;
	private int _d;
	private int _stride;
	private int _pad;

	private int[] _mapx;
	private int[] _mapy;


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

		_mapx = new int[this.outLength()];
		_mapy = new int[this.outLength()];

	}


	public Volume forward(Volume c) {

		this.input = c;
		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int n = 0; // a counter for switches
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outH(); y += this._stride, ay++) {
					double a = -999999999;
					int px = -1;
					int py = -1;
					for (int fx = 0; fx < _w; fx++) {
						for (int fy = 0; fy < _h; fy++) {
							int oy = y + fy;
							int ox = x + fx;
							if (oy >= 0 && oy < c.height() && ox >= 0 && ox < c.width()) {
								double v = c.get(ox, oy, d);
								if (v > a) {
									a = v;
									px = ox;
									py = oy;
								}
							}
						}
					}
					this._mapx[n] = px;
					this._mapy[n] = py;
					n++;
					A.set(ax, ay, d, a);
				}
			}
		}

		this.output = A;
		return this.output;
	}


	public void backward() {
		Volume V = this.input;
		V.dW = new double[V.W.length];
		Volume A = this.output;
		int n = 0;
		for (int d = 0; d < this.outD(); d++) {
			int x = -this._pad;
			int y = -this._pad;
			for (int ax = 0; ax < this.outW(); x += this._stride, ax++) {
				y = -this._pad;
				for (int ay = 0; ay < this.outW(); y += this._stride, ay++) {
					double grad = this.output.getGrad(ax, ay, d);
					try {
					V.addGrad(_mapx[n], _mapy[n], d, grad);
					
					n++;
					}
					catch (Exception ex) {
						Console.writeLine(ex.getMessage());
					}
				}
			}
		}
	}

}