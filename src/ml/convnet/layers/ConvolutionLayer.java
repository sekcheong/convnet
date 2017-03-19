package ml.convnet.layers;

public class ConvolutionLayer extends Layer {	
	private int _width;  //filter width
	private int _height; //filter height
	private int _depth;  //filter depth
	private int _stride;
	private int _padding;
	private Layer _activation;
	
	//sx:3, sy: 4, pad:1, filters:10, stride:1, activation:'relu'
	public ConvolutionLayer(int width, int height, int filters, int stride, int padding, Layer activation) {
		_width = width;
		_height = height;
		_depth = filters;
		_stride = stride;
		_padding = padding;
		_activation = activation;
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
