package ml.convnet.layer;

import ml.convnet.Cube;

public class ConvolutionLayer extends Layer {

	private Cube[] _filters;
	private int _stride;
	int _pad;
	int _filterW;
	int _filterH;


	public ConvolutionLayer(Layer prev, int filterW, int filterH, int filterD, int stride, int pad, double bias) {
		super(prev);
		int t;

		_filterW = filterW;
		_filterH = filterH;

		_stride = stride;
		_pad = pad;

		if (_stride <= 0) _stride = 1;
		if (filterH == 0) _filterH = _filterW;
		
		this.inW(prev.outW()).inH(prev.outH()).inD(prev.outD());
		
		t = (int) Math.floor((double) (this.inW() + pad * 2 - _filterW) / _stride + 1);
		this.outW(t);

		t = (int) Math.floor((double) (this.inH() + pad * 2 - _filterH) / _stride + 1);
		this.outH(t);

		this.outD(filterD);

		_filters = new Cube[this.outD()];
		for (int i = 0; i < _filters.length; i++) {
			_filters[i] = new Cube(_filterW, _filterH, this.inD());
		}

		this.bias = bias;
		this.biases = new Cube(1, 1, this.outD(), bias);
		this.type = LayerType.convolution;
	}


	public Cube forward(Cube v) {

		this.input = v;
		Cube out = new Cube(_filterW, _filterH, this.outD(), 0.0);

		int w = v.width();
		int h = v.height();
		int stride = _stride;

		for (int d = 0; d < this.outD(); d++) {
			Cube filter = _filters[d];
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
		this.output = out;
		return out;
	}


	public void backward() {

		Cube V = this.input;
		V.dW = new double[V.W.length]; // zero out gradient wrt bottom data, we're about to fill it
		
		int V_sx = V.width();
		int V_sy = V.height();
		int xy_stride = _stride;

		for (int d = 0; d < this.outD(); d++) {
			Cube f = _filters[d];
			int x = -this._pad;
			int y = -this._pad;
			for (int ay = 0; ay < this.outH(); y += xy_stride, ay++) {
				x = -this._pad;
				for (int ax = 0; ax < this.outW(); x += xy_stride, ax++) {
					// convolve centered at this particular location
					double chainGrad = this.output.dW[Cube.index(ax, ay, d)]; // gradient from above, from chain rule
					for (int fy = 0; fy < f.height(); fy++) {
						int oy = y + fy; // coordinates in the original input array coordinates
						for (int fx = 0; fx < f.width(); fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.depth(); fd++) {
									// avoid function call overhead (x2) for efficiency, compromise modularity :(
									int ix1 = ((V_sx * oy) + ox) * V.depth() + fd;
									int ix2 = ((f.width() * fy) + fx) * f.depth() + fd;
									f.dW[ix2] += V.W[ix1] * chainGrad;
									V.dW[ix1] += f.W[ix2] * chainGrad;
								}
							}
						}
					}
					this.biases.dW[d] += chainGrad;
				}
			}
		}
	}
	
	public double[][][] getResponse() {
		int n = this.outD();
		double[][][] res = new double[n + 1][2][];
		for (int i = 0; i < n; i++) {
			res[i][0] = _filters[i].W;
			res[i][1] = _filters[i].dW;
		}
		res[n][0] = this.biases.W;
		res[n][1] = this.biases.dW;
		return res;
	}

}
