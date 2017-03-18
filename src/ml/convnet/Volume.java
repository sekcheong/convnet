package ml.convnet;

public class Volume {
	
	private int _width;
	private int _height;
	private int _depth;
	private double[] _w;
	private double[] _dw;
	
	public Volume(int width, int height, int depth) {
		_width=width;
		_height=height;
		_depth=depth;
		_w = new double[_width*_height*_depth];
		_dw = new double[_w.length];
	}
	
	

}

