package ml.data;

public class DataSet<T> {
	T[] _examples;


	public DataSet(T[] examples) {
		_examples = examples;
	}


	public void shuffle() {
		for (int i = _examples.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			T t = _examples[i];
			_examples[i] = _examples[j];
			_examples[j] = t;
		}
	}


	public int count() {
		return _examples.length;
	}


	public T get(int index) {
		return _examples[index];
	}

}
