package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.convnet.Volume;
import ml.data.Example;

public class SGDTrainer extends Trainer {

	private double _rate;
	private double _momentum;
	private double _decayL1;
	private double _decayL2;
	private double _decayLossL1;
	private double _decayLossL2;
	private double _loss = 0.0;
	private int _batchSize;


	public SGDTrainer(double learningRate, int batchSize, double momentum, double decayL1, double decayL2) {
		_rate = learningRate;
		_momentum = momentum;
		_decayL1 = decayL1;
		_decayL2 = decayL2;
		_batchSize = batchSize;
	}
	

	@Override
	protected void trainOneExample(ConvNet net, double[] x, double[] y) {
		double[][] gsum = null;

		_decayLossL1 = 0.0;
		_decayLossL2 = 0.0;
		_loss = 0.0;

		this.net().forward(x);
		_loss = this.net().backward(y);		

		if ((this.iteration() % _batchSize) == 0) {

			//get the network weights and gradients
			Volume[] r = this.net().response();

			//for momentum we need to use 
			if (_momentum > 0 && gsum == null) {
				gsum = new double[r.length][];
				for (int i = 0; i < r.length; i++) {
					gsum[i] = new double[r[i].dW.length];
				}
			}


			for (int i = 0; i < r.length; i++) {

				double[] w = r[i].W;
				double[] g = r[i].dW;

				for (int j = 0; j < w.length; j++) {
					// accumulate weight decay loss
					_decayLossL1 += _decayL1 * Math.abs(w[j]);
					_decayLossL2 += _decayL2 * w[j] * w[j] / 2;

					double gradL1 = _decayL1 * (w[j] > 0 ? 1 : -1);
					double gradL2 = _decayL2 * (w[j]);

					// raw batch gradient
					double gij = (gradL1 + gradL2 + g[j]) / _batchSize;
					double[] gsum_i = gsum[i];

					if (_momentum > 0.0) {
						double dx = _momentum * gsum_i[j] - _rate * gij;
						gsum_i[j] = dx; // back this up for next iteration of momentum
						w[j] += dx;     // apply corrected gradient
					}
					else {
						w[j] += -_rate * gij;
					}
					g[j] = 0.0;
				}

			}

		}

	}

}