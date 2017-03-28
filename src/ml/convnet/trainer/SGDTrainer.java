package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.convnet.Volume;

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


		_decayLossL1 = 0.0;
		_decayLossL2 = 0.0;
		_loss = 0.0;

		this.net().forward(x);
		_loss = this.net().backward(y);

		if ((this.step() % _batchSize) == 0) {

			Volume[] r = this.net().response();

			double[][] gs = new double[r.length][];
			for (int i = 0; i < r.length; i++) {
				gs[i] = new double[r[i].dW.length];
			}

			for (int i = 0; i < r.length; i++) {

				double[] w = r[i].W;
				double[] g = r[i].dW;

				for (int j = 0; j < w.length; j++) {

					_decayLossL1 += _decayL1 * Math.abs(w[j]);
					_decayLossL2 += _decayL2 * w[j] * w[j] / 2;

					double gradL1 = _decayL1 * (w[j] > 0 ? 1 : -1);
					double gradL2 = _decayL2 * (w[j]);

					double delta = (gradL1 + gradL2 + g[j]) / _batchSize;
					double[] gs_i = gs[i];

					if (_momentum > 0.0) {
						double dx = _momentum * gs_i[j] - _rate * delta;
						gs_i[j] = dx;
						w[j] += dx;
					}
					else {
						w[j] += -_rate * delta;
					}
					g[j] = 0.0;
				}

			}

		}

	}


	public double costLoss() {
		return _loss;
	}


	public double decayLossL1() {
		return _decayLossL1;
	}


	public double decayLossL2() {
		return _decayLossL2;
	}

}