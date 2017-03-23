package ml.io;

import ml.data.DataSet;

public abstract class DataReader<T> {

	public abstract void read();

	public abstract DataSet<T> dataSet();
}
