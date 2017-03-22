package ml.convnet.learner;

import ml.convnet.Volume;

public class SGDLearner extends Learner {
	private double _rate;
	private double _momentum;
	private double _decayL1;
	private double _decayL2;
	private int _batchSize;


	public SGDLearner(double learningRate, int batchSize, double momentum, double decayL1, double decayL2) {
		_rate = learningRate;
		_momentum = momentum;
		_decayL1 = decayL1;
		_decayL2 = decayL2;
		_batchSize = batchSize;
	}


	@Override
	public void train(double[] x, double[] y) {
		double loss = 0.0;
		double decayLossL1 = 0.0;
		double decayLossL2 = 0.0;
		double[][] gradSum = null;

		_net.forward(x);
		loss = _net.backward(y);

		this.incIteration();

		if ((this.iteration() % _batchSize) == 0) {

			Volume[] r = _net.response();

			if (_momentum > 0 && gradSum == null) {
				gradSum = new double[r.length][];
				for (int i = 0; i < r.length; i++) {
					gradSum[i] = new double[r[i].dW.length];
				}
			}

			for (int i = 0; i < r.length; i++) {

				double[] p = r[i].W;
				double[] g = r[i].dW;

				for (int j = 0; j < p.length; j++) {

					// accumulate weight decay loss
					decayLossL2 += _decayL2 * p[j] * p[j] / 2;
					decayLossL1 += _decayL1 * Math.abs(p[j]);

					double gradL1 = _decayL1 * (p[j] > 0 ? 1 : -1);
					double gradL2 = _decayL2 * (p[j]);

					double gij = (gradL1 + gradL2 + g[j]) / _batchSize; // raw batch gradient
					double gsumi[] = gradSum[i];

					if (_momentum > 0.0) {
						double dx = _momentum * gsumi[j] - _rate * gij;
						// back this up for next iteration of momentum
						gsumi[j] = dx;
						// apply corrected gradient
						p[j] += dx;
					}
					else {
						p[j] += -_rate * gij;
					}

					g[j] = 0.0;
				}

			}

		}

	}

}