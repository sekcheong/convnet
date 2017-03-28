package ml.convnet.layer;

import ml.convnet.Volume;

public class Convolution extends Layer {

	private Volume[] _filters;
	private int _stride;
	private int _pad;

	private int _filterW;
	private int _filterH;
	private int _filterD;


	public Convolution(int filterW, int filterH, int filterD, int stride, int pad, double bias) {
		this.type = LayerType.convolution;
		_filterW = filterW;
		_filterH = (filterH == 0) ? filterW : filterH;
		_filterD = filterD;
		_stride = (stride == 0) ? 1 : stride;
		_pad = pad;
		this.bias = bias;
	}


	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		int w = (int) Math.floor((double) (this.inW() + _pad * 2 - _filterW) / _stride + 1);
		this.outW(w);

		int h = (int) Math.floor((double) (this.inH() + _pad * 2 - _filterH) / _stride + 1);
		this.outH(h);

		this.outD(_filterD);

		_filters = new Volume[_filterD];
		for (int i = 0; i < _filters.length; i++) {
			_filters[i] = new Volume(_filterW, _filterH, this.inD());
		}

		this.biases = new Volume(1, 1, this.outD(), this.bias);
	}


	public Volume forward(Volume c) {

		this.input = c;
		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int inW = c.width();
		int inH = c.height();

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outH(); y += _stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += _stride, ax++) {
					double a = 0.0;
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy;
						for (int fx = 0; fx < f.height(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < inH && ox >= 0 && ox < inW) {
								for (int fd = 0; fd < f.depth(); fd++) {
									a += f.W[((f.width() * fy) + fx) * f.depth() + fd] * c.W[((inW * oy) + ox) * c.depth() + fd];
								}
							}
						}
					}
					a += this.biases.W[d];
					A.set(ax, ay, d, a);
				}
			}
		}
		this.output = A;
		return this.output;
	}


	public void backward() {
		Volume in = this.input;
		in.dW = new double[in.W.length];

		int inW = in.width();
		int inH = in.height();

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outW(); y += _stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += _stride, ax++) {
					double grad = this.output.getGrad(ax, ay, d);
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy;
						for (int fx = 0; fx < f.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < inH && ox >= 0 && ox < inW) {
								for (int fd = 0; fd < f.depth(); fd++) {
									int ix1 = ((inW * oy) + ox) * in.depth() + fd;
									int ix2 = ((f.width() * fy) + fx) * f.depth() + fd;
									f.dW[ix2] += in.W[ix1] * grad;
									in.dW[ix1] += f.W[ix2] * grad;
								}
							}
						}
					}
					this.biases.dW[d] += grad;
				}
			}
		}
	}


	public Volume[] response() {
		Volume[] ret = new Volume[_filters.length + 1];
		for (int i = 0; i < _filters.length; i++) {
			ret[i] = _filters[i];
		}
		ret[_filters.length] = this.biases;
		return ret;
	}

}
