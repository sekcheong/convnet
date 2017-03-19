package ml.convnet.layers;

import java.util.ArrayList;
import java.util.List;

import ml.convnet.Cube;
import ml.convnet.CubeSize;

public class ConvolutionLayer extends Layer {

	private List<Cube> _filters = new ArrayList<Cube>();
	private CubeSize _filterSize;
	private int _stride;
	private int _padding;

	public ConvolutionLayer(int w, int h, int d, int filterW, int filterD, int stride, int padding) {
		initialize(w, h, d, filterW, filterW, filterD, stride, padding);
	}


	public ConvolutionLayer(int w, int h, int d, int filterW, int filterH, int filterD, int stride, int padding) {
		initialize(w, h, d, filterW, filterH, filterD, stride, padding);
	}


	private void initialize(int w, int h, int d, int filterW, int filterH, int filterD, int stride, int padding) {

		this.inputSize = new CubeSize(w, h, d);
		
		int outW = (int) Math.floor((w + padding * 2 - filterW) / (double) stride + 1);
		int outH = (int) Math.floor((h + padding * 2 - filterH) / (double) stride + 1);
		int outD = filterD;
		this.outputSize = new CubeSize(outW, outH, outD);

		if (filterH == 0) filterH = filterW;
		_filterSize = new CubeSize(filterW, filterH, filterD);
		
		for (int i = 0; i < outD; i++) {
			_filters.add(new Cube(filterW, filterH, this.inputSize.d));
		}

		this.type = LayerType.convolution;
	}


	@Override
	public Cube forward(Cube x) {
		this.input = x;
		Cube out = new Cube(this.outputSize.w, this.outputSize.h, this.outputSize.d, 0);

		// var xy_stride = this.stride |0;
		
		for (int d=0; d<this.outputSize.d; d++) {
			Cube f = _filters.get(d);
			int i = -_padding;
			int j = -_padding;
			
			for (int ay = 0; ay < this.outputSize.h; j += _stride, ay++) { // xy_stride
				j = -_padding;
				for (int ax = 0; ax < this.outputSize.w; j += _stride, ax++) { // xy_stride

				}
			}

		}
		
		// for(var d=0;d<this.out_depth;d++) {
		// var f = this.filters[d];
		// var x = -this.pad |0;
		// var y = -this.pad |0;
		// for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) { // xy_stride
		// x = -this.pad |0;
		// for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) { // xy_stride
		//
		// // convolve centered at this particular location
		// var a = 0.0;
		// for(var fy=0;fy<f.sy;fy++) {
		// var oy = y+fy; // coordinates in the original input array coordinates
		// for(var fx=0;fx<f.sx;fx++) {
		// var ox = x+fx;
		// if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
		// for(var fd=0;fd<f.depth;fd++) {
		// // avoid function call overhead (x2) for efficiency, compromise modularity :(
		// a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
		// }
		// }
		// }
		// }
		// a += this.biases.w[d];
		// A.set(ax, ay, d, a);
		// }
		// }
		// }
		// this.out_act = A;
		// return this.out_act;
		return null;
	}


	@Override
	public void backward() {
		// TODO Auto-generated method stub

	}


	@Override
	public Cube backward(Cube y) {
		// TODO Auto-generated method stub
		return null;
	}

}
