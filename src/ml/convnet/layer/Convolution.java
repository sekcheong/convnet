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


	public Volume forward(Volume V) {

		this.input = V;
		Volume A = new Volume(this.outW(), this.outH(), this.outD(), 0.0);

		int V_sx = V.width();
		int V_sy = V.height();
		int xy_stride = _stride;

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outH(); y += xy_stride, ay++) { // xy_stride
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += xy_stride, ax++) { // xy_stride

					// convolve centered at this particular location
					double a = 0.0;
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy; // coordinates in the original input array coordinates
						for (int fx = 0; fx < f.height(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.depth(); fd++) {
									// avoid function call overhead (x2) for efficiency, compromise modularity :(
									a += f.W[((f.width() * fy) + fx) * f.depth() + fd] * V.W[((V_sx * oy) + ox) * V.depth() + fd];
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
		Volume V = this.input;
		V.dW = new double[V.W.length]; // zero out gradient wrt bottom data, we're about to fill it

		int V_sx = V.width();
		int V_sy = V.height();
		int xy_stride = _stride;

		for (int d = 0; d < this.outD(); d++) {
			Volume f = this._filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outW(); y += xy_stride, ay++) { // xy_stride
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += xy_stride, ax++) { // xy_stride
					// convolve centered at this particular location
					// gradient from above, from chain rule
					double chain_grad = this.output.getGrad(ax, ay, d);
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy; // coordinates in the original input array coordinates
						for (int fx = 0; fx < f.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.depth(); fd++) {
									// avoid function call overhead (x2) for efficiency, compromise modularity :(
									int ix1 = ((V_sx * oy) + ox) * V.depth() + fd;
									int ix2 = ((f.width() * fy) + fx) * f.depth() + fd;
									f.dW[ix2] += V.W[ix1] * chain_grad;
									V.dW[ix1] += f.W[ix2] * chain_grad;
								}
							}
						}
					}
					this.biases.dW[d] += chain_grad;
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
