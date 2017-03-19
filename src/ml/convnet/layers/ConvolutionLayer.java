package ml.convnet.layers;

public class ConvolutionLayer extends Layer {
	private int _width; // filter width
	private int _height; // filter height
	private int _depth; // filter depth
	private int _stride;
	private int _padding;
	private Layer _activation;


	public ConvolutionLayer(int width, int filters, int stride, int padding, Layer activation) {
		initLayer(width, 0, filters, stride, padding, activation);
	}


	public ConvolutionLayer(int width, int height, int filters, int stride, int padding, Layer activation) {
		initLayer(width, height, filters, stride, padding, activation);
	}


	private void initLayer(int width, int height, int filters, int stride, int padding, Layer activation) {
		_width = width;
		_height = height>0 ?  height : width;
		_depth = filters;
		_stride = stride;
		_padding = padding;
		_activation = activation;

//		_outWidth = Math.floor((this.inputWidth() + padding * 2 - width) / stride + 1);
//		this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
		this.type(LayerType.convolution);
	}


	@Override
	void forward() {
		// TODO Auto-generated method stub

	}


	@Override
	void backward() {
		// TODO Auto-generated method stub

	}

}
