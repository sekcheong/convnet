package ml.convnet.layer;

import ml.convnet.Cube;

public class ConvolutionLayer extends Layer {

	private Cube[] _filters;
	private int _stride;
	int _outW;
	int _outH;
	int _outD;
	int _inW;
	int _inH;
	int _inD;
	int _pad;
	int _w;
	int _h;


	public ConvolutionLayer(int inW, int inH, int inD, int fW, int fH, int fD, int stride, int pad, double bias) {
		_outD = fD;
		_w = fW;
		_h = fH;
		_inD = inD;
		_inW = inW;
		_inH = inH;
		_stride = stride;
		_pad = pad;

		if (_stride <= 0) _stride = 1;
		if (fH == 0) _h = _w;

		_outW = (int) Math.floor((double) (_inW + pad * 2 - _w) / _stride + 1);
		_outH = (int) Math.floor((double) (_inH + pad * 2 - _h) / _stride + 1);

		_filters = new Cube[_outD];
		for (int i = 0; i < _outD; i++) {
			_filters[i] = new Cube(_w, _h, _inD);
		}
		this.biases = new Cube(1, 1, _outD, bias);

		this.type = LayerType.convolution;
	}


	public Cube forward(Cube V) {

		this.input = V;
		Cube A = new Cube(_w, _h, _outD, 0.0);

		int V_sx = V.dim().w;
		int V_sy = V.dim().h;
		int xy_stride = _stride;

		for (int d = 0; d < _outD; d++) {
			Cube f = _filters[d];
			int x = -_pad;
			int y = -_pad;
			for (int ay = 0; ay < _outH; y += xy_stride, ay++) {
				x = -_pad;
				for (int ax = 0; ax < _outW; x += xy_stride, ax++) {
					// convolve centered at this particular location
					double a = 0.0;
					for (int fy = 0; fy < f.dim().h; fy++) {
						int oy = y + fy; // coordinates in the original input array coordinates
						for (int fx = 0; fx < f.dim().w; fx++) {
							int ox = x + fx;
							if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
								for (int fd = 0; fd < f.dim().d; fd++) {
									a += f.get(fx, fy, fd) * V.get(ox, oy, fd);
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
		return A;
	}

}
