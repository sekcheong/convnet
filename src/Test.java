

public class Test {
	public static void main(String[] args) {
		int[] a = new int[10];
		for (int i=0; i<a.length; i++) a[i]=i;
		for (int i = a.length - 1; i >= 1; i--) {
			int j = (int) (Math.random() * i);
			int t = a[i];
			a[i] = a[j];
			a[j] = t;
		}
		for (int i=0; i<a.length; i++) System.out.println(a[i]);
	}
}
