package ml.convnet.layer;

public class PoolLayer extends Layer {
	private int _w;
	private int _h;
	private int _d;
	private int _stride;
	private int _pad;
	private int[][] _mask;

	public PoolLayer(Layer prev, int w, int h, int stride, int pad) {
		super(prev);
		
		int t;
		_w = w;
		_h = (h == 0) ? _w : _h;
		_stride = stride;
		_pad = pad;

		this.inW(prev.outW()).inH(prev.outH()).inD(prev.outD());

		this.outD(this.inD());

		t = (int) Math.floor((double) (this.inW() + _pad * 2 - _w) / _stride + 1);
		this.outW(t);

		t = (int) Math.floor((double) (this.inH() + _pad * 2 - _h) / _stride + 1);
		this.outH(t);

		// stores mask for x, y coordinates for where the max comes from, for each output neuron
		_mask = new int[this.outLength()][2];
	
		// this.switchx = global.zeros(this.out_sx*this.out_sy*this.out_depth);
		// this.switchy = global.zeros(this.out_sx*this.out_sy*this.out_depth);

		this.type = LayerType.pool;
	}

}