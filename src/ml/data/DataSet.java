package ml.data;

import java.util.List;

public class DataSet {

	private Example[] _examples;


	public DataSet() {
		_examples = new Example[0];
	}
	
	public DataSet(Example[] examples) {
		_examples = new Example[examples.length];
		for (int i = 0; i < examples.length; i++) {
			_examples[i] = examples[i];
		}
	}


	public DataSet(List<Example> examples) {
		_examples = new Example[examples.size()];
		for (int i = 0; i < examples.size(); i++) {
			_examples[i] = examples.get(i);
		}
	}


	public void shuffle() {
		for (int i = _examples.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			Example t = _examples[i];
			_examples[i] = _examples[j];
			_examples[j] = t;
		}
	}


	public DataSet[] split() {
		return null;
	}


	public int count() {
		return _examples.length;
	}


	public Example get(int index) {
		return _examples[index];
	}


	public Example[] examples() {
		return _examples;
	}

}
