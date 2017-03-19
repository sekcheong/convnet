package ml.convnet;

public final class CubeSize {
	public CubeSize(int w, int h, int d) {
		this.w = w;
		this.h = h;
		this.d = d;
	}

	public int w;
	public int h;
	public int d;

	public int size() {
		return w * h * d;
	}


	public int index(int x, int y, int z) {
		return ((w * h * y) + x) * d + d + z;
	}

}
