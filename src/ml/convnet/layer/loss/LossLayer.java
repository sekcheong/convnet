package ml.convnet.layer.loss;

import ml.convnet.layer.Layer;

public class LossLayer extends Layer {

	public void connect(Layer l) {

		this.inW(l.outW());
		this.inH(l.outH());
		this.inD(l.outD());

		this.outW(1);
		this.outH(1);
		this.outD(this.inLength());
	}

}
