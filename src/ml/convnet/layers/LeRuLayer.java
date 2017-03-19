package ml.convnet.layers;

import ml.convnet.Cube;

public class LeRuLayer extends Layer {

	@Override
	public Cube forward(Cube v) {
		this.input = v;
		Cube out = new Cube(v);
		double[] w = out.W;
		for (int i = 0; i < w.length; i++) {
			if (w[i] < 0) w[i] = 0;
		}
		this.output = out;
		return this.output;
	}


	@Override
	public void backward() {
		Cube vi = this.input;
		Cube vo = this.output;
		vi.dW = Cube.zeros(vi.W.length);
		for (int i = 0; i < vi.W.length; i++) {
			if (vo.W[i] <= 0) vi.dW[i] = 0;
			else vi.dW[i] = vo.dW[i];
		}
	}


	@Override
	public Cube backward(Cube v) {
		// TODO Auto-generated method stub
		return null;
	}

}
