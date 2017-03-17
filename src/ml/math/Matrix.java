package ml.math;

interface Function {
	double compute(double a);
}

public class Matrix {
	private int _rows;
	private int _cols;
	double[][] A;


	public Matrix(int rows, int cols) {
		A = new double[rows][cols];
	}


	public Matrix(Matrix src) {
		A = copy(src.A);
		this._cols = src.cols();
		this._rows = src.rows();
	}


	public Matrix(double[][] src) {
		A = copy(src);
		this._rows = A.length;
		this._cols = A[0].length;
	}


	public int rows() {
		return _rows;
	}


	public int cols() {
		return _cols;
	}


	public Matrix plus(Matrix b) {
		return this;
	}


	public Matrix plusEqual(Matrix b) {
		return this;
	}


	public Matrix transpose() {
		return new Matrix(transpose(this.A));
	}


	public static Matrix identity(int rows, int cols) {
		Matrix b = new Matrix(rows, cols);
		return identity(b);
	}


	public static Matrix identity(Matrix m) {
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < m.cols(); j++) {
				m.A[i][j] = (i == j ? 1.0 : 0.0);
			}
		}
		return m;
	}


	public static Matrix copy(Matrix src) {
		return new Matrix(src);
	}


	private static double[][] transpose(double[][] a) {
		int cols = a[0].length;
		double[][] b = new double[cols][a.length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				b[j][i] = a[i][j];
			}
		}
		return b;
	}


	private static void plus(double[][] a, double[][] b) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] += b[i][j];
			}
		}
	}


	private static void minus(double[][] a, double[][] b) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] -= b[i][j];
			}
		}
	}


	private static void div(double[][] a, double[][] b) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] /= b[i][j];
			}
		}
	}


	private static void times(double[][] a, double[][] b) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] *= b[i][j];
			}
		}
	}


	private static double[][] dot(double[][] a, double[][] b) {
		if (a[0].length != b[0].length) {
			throw new IllegalArgumentException("Matrix inner dimensions must agree.");
		}

		int rows = a.length;
		int cols = b.length;
		int m = a[0].length;

		double[][] c = new double[rows][cols];
		double sum = 0;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				sum = 0;
				for (int k = 0; k < m; k++) {
					sum = sum + a[i][k] * b[k][j];
				}
				c[i][j] = sum;
			}
		}
		return c;
	}


	private static void apply(double[][] a, Function f) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] = f.compute(a[i][j]);
			}
		}
	}


	private static void fill(double[][] a, double b) {
		int cols = a[0].length;
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < cols; j++) {
				a[i][j] = b;
			}
		}
	}


	private static double[][] copy(double[][] src) {
		int cols = src[0].length;
		double[][] dest = new double[src.length][cols];
		for (int i = 0; i < src.length; i++) {
			for (int j = 0; j < cols; j++) {
				dest[i][j] = src[i][j];
			}
		}
		return dest;
	}


	private void checkDim(Matrix b) {
		if (b._cols != this._cols || b._rows != this._cols) {
			throw new IllegalArgumentException("Matrix dimensions must agree.");
		}
	}

}

//
//
//
// private static void fft2(double[][] x) {
// int h = x.length;
// int w = x[0].length;
// double[][][] y = new double[3][h][w];
// double[][] r = y[0];
// double[][] i = y[1];
// double[][] a = y[2];
// for (int yw = 0; yw < h; yw++) {
// for (int xw = 0; xw < w; xw++) {
// for (int ys = 0; ys < h; ys++) {
// for (int xs = 0; xs < w; xs++) {
// r[yw][xw] += (x[ys][xs] * Math.cos(2 * Math.PI * ((1.0 * xw * xs / w) + (1.0 * yw * ys / h)))) / Math.sqrt(w * h);
// i[yw][xw] -= (x[ys][xs] * Math.sin(2 * Math.PI * ((1.0 * xw * xs / w) + (1.0 * yw * ys / h)))) / Math.sqrt(w * h);
// a[yw][xw] = Math.sqrt(r[yw][xw] * r[yw][xw] + i[yw][xw] * i[yw][xw]);
// }
// }
// }
// }
// }
//
//
// private static void ifft2(double[][] x) {
// int h = x.length;
// int w = x[0].length;
// double[][][] y = new double[3][h][w];
// double[][] r = y[0];
// double[][] i = y[1];
// double[][] a = y[2];
// for (int yw = 0; yw < h; yw++) {
// for (int xw = 0; xw < w; xw++) {
// for (int ys = 0; ys < h; ys++) {
// for (int xs = 0; xs < w; xs++) {
// r[yw][xw] += (x[ys][xs] * Math.cos(2 * Math.PI * ((1.0 * xw * xs / w) + (1.0 * yw * ys / h)))) / Math.sqrt(w * h);
// i[yw][xw] -= (x[ys][xs] * Math.sin(2 * Math.PI * ((1.0 * xw * xs / w) + (1.0 * yw * ys / h)))) / Math.sqrt(w * h);
// a[yw][xw] = Math.sqrt(r[yw][xw] * r[yw][xw] + i[yw][xw] * i[yw][xw]);
// }
// }
// }
// }
// }
