package ml.convnet.layer;

import ml.convnet.Volume;

public class Convolution extends Layer {

	private Volume[] _filters;
	private int _stride;
	int _pad;
	int _filterW;
	int _filterH;
	int _filterD;


	public Convolution(int filterW, int filterH, int filterD, int stride, int pad, double bias) {		
		this.type = LayerType.convolution;		
		_filterW = filterW;
		_filterH = filterH;
		_filterD = filterD;
		_stride = stride;
		_pad = pad;
		this.bias=bias;
	}


	public void connect(Layer l) {		
		
		if (_stride <= 0) _stride = 1;
		if (_filterH == 0) _filterH = _filterW;

		this.inW(l.outW()).inH(l.outH()).inD(l.outD());

		int t;
		t = (int) Math.floor((double) (this.inW() + _pad * 2 - _filterW) / _stride + 1);
		this.outW(t);

		t = (int) Math.floor((double) (this.inH() + _pad * 2 - _filterH) / _stride + 1);
		this.outH(t);

		this.outD(_filterD);

		_filters = new Volume[this.outD()];
		for (int i = 0; i < _filters.length; i++) {
			_filters[i] = new Volume(_filterW, _filterH, this.inD());
		}

		this.biases = new Volume(1, 1, this.outD(), this.bias);
	}
	

	public Volume forward(Volume v) {

		this.input = v;
		Volume out = new Volume(_filterW, _filterH, this.outD(), 0.0);
		this.output = out;

		int w = v.width();
		int h = v.height();
		int stride = _stride;

		for (int d = 0; d < this.outD(); d++) {
			Volume filter = _filters[d];
			int x = -_pad;
			int y = -_pad;
			for (int ay = 0; ay < this.outH(); y += stride, ay++) {
				x = -_pad;
				for (int ax = 0; ax < this.outW(); x += stride, ax++) {
					double a = 0.0;
					for (int fy = 0; fy < filter.height(); fy++) {
						int oy = y + fy;
						for (int fx = 0; fx < filter.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < h && ox >= 0 && ox < w) {
								for (int fd = 0; fd < filter.depth(); fd++) {
									a += filter.get(fx, fy, fd) * v.get(ox, oy, fd);
								}
							}
						}
					}
					a += this.biases.W[d];
					out.set(ax, ay, d, a);
				}
			}
		}

		return out;
	}


	public void backward() {

		Volume in = this.input;
		in.dW = new double[in.W.length]; // zero out gradient wrt bottom data,
		// we're about to fill it
		int V_sx = in.width();
		int V_sy = in.height();
		int xy_stride = _stride;

		for (int d = 0; d < _filters.length; d++) {
			Volume f = _filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outH(); y += xy_stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += xy_stride, ax++) {
					// convolve centered at this particular location
					// gradient from above, from chain rule
					double chainGrad = this.output.getGrad(ax, ay, d);
					for (int fy = 0; fy < f.height(); fy++) {
						// coordinates in the original input array coordinates
						int oy = y + fy;
						for (int fx = 0; fx < f.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.depth(); fd++) {
									int ix1 = ((V_sx * oy) + ox) * in.depth() + fd;
									int ix2 = ((f.width() * fy) + fx) * f.depth() + fd;
									f.dW[ix2] += in.W[ix1] * chainGrad;
									in.dW[ix1] += f.W[ix2] * chainGrad;
								}
							}
						}
					}
					this.biases.dW[d] += chainGrad;
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
