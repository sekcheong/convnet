package ml.utils.tracing;

public class StopWatch {

	private long _start;
	private long _elapsedTime;


	public StopWatch() {}


	public void start() {
		_start = System.nanoTime();
	}


	public void stop() {
		_elapsedTime = System.nanoTime() - _start;
	}


	public double elapsedTime() {
		return ((double) _elapsedTime) * 10e-10;
	}
}
