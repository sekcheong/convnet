package ml.convnet.data;

import ml.convnet.Volume;

public class Example {
	public Volume x;
	public Volume y;


	public Example(double[] x, double[] y) {
		this.x = new Volume(x);
		this.y = new Volume(y);
	}


	public Example(Volume x, Volume y) {
		this.x = x;
		this.y = y;
	}

}