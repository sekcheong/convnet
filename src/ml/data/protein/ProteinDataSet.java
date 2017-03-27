package ml.data.protein;

import ml.data.DataSet;
import ml.data.Example;

public class ProteinDataSet extends DataSet {

	private Example[] _train;
	private Example[] _tune;
	private Example[] _test;


	public ProteinDataSet(Example[] train, Example[] tune, Example[] test) {
		super(train);
		_train = train;
		_tune = tune;
		_test = test;
	}


	public DataSet[] split() {
		DataSet[] ds = new DataSet[3];
		ds[0] = new DataSet(_train);
		ds[1] = new DataSet(_tune);
		ds[2] = new DataSet(_test);
		return ds;
	}

}
