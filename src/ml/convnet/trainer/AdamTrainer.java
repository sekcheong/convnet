package ml.convnet.trainer;

import ml.convnet.ConvNet;
import ml.convnet.Volume;

public class AdamTrainer extends Trainer {

	private double _rate;

	private double _momentum;

	private double _decayL1;

	private double _decayL2;

	private double _decayLossL1;

	private double _decayLossL2;

	private double _loss = 0.0;

	private int _batchSize;

	private double _beta1;

	private double _beta2;

	private double _eps;


	public AdamTrainer(double learningRate, int batchSize, double momentum, double decayL1, double decayL2, double beta1, double beta2, double eps) {
		_rate = learningRate;
		_momentum = momentum;
		_decayL1 = decayL1;
		_decayL2 = decayL2;
		_beta1 = beta1;
		_beta2 = beta2;
		_eps = eps;
		_batchSize = batchSize;
	}


	@Override
	protected void trainOneExample(ConvNet net, double[] x, double[] y) {
		double[][] gsum = null;
		double[][] xsum = null;
		_decayLossL1 = 0.0;
		_decayLossL2 = 0.0;
		_loss = 0.0;

		this.net()
				.forward(x);
		_loss = this.net()
				.backward(y);

		if ((this.step() % _batchSize) == 0) {

			// get the network weights and gradients
			Volume[] r = this.net()
					.response();

			gsum = new double[r.length][];
			xsum = new double[r.length][];
			for (int i = 0; i < r.length; i++) {
				gsum[i] = new double[r[i].dW.length];
				xsum[i] = new double[r[i].dW.length];
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
					double[] gsumi = gsum[i];
					double[] xsumi = xsum[i];
					// adam update
					gsumi[j] = gsumi[j] * this._beta1 + (1 - this._beta1) * gij; // update biased first moment estimate
					xsumi[j] = xsumi[j] * this._beta2 + (1 - this._beta2) * gij * gij; // update biased second moment estimate
					double biasCorr1 = gsumi[j] * (1 - Math.pow(this._beta1, this.step())); // correct bias first moment estimate
					double biasCorr2 = xsumi[j] * (1 - Math.pow(this._beta2, this.step())); // correct bias second moment estimate
					double dx = -this._rate * biasCorr1 / (Math.sqrt(biasCorr2) + this._eps);
					w[j] += dx;
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