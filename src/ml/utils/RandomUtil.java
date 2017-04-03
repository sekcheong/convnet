package ml.utils;

import java.util.Random;

public  class RandomUtil {
	private static Random  _r = new Random(838*838);
	
	public static double nextDouble() {
		return _r.nextDouble();
	}
	
	public static double nextGaussian() {
		return _r.nextGaussian();
	}
}
