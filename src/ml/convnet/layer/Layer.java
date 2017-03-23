package ml.convnet.layer;

import ml.convnet.ConvNet;
import ml.convnet.Volume;

public abstract class Layer {
	public ConvNet net;
	public LayerType type;
	public Volume input;
	public Volume output;
	public Volume biases;
	public int[][] sizes = new int[2][3];

	private Layer _next = null;
	private Layer _last = null;

	public double bias;
	public int index;
	private ConvNet _net;


	public Layer() {

	}


	public Volume forward(double[] x) {
		return null;
	}
	
	public Volume forward(Volume x) {
		return null;
	}


	public void backward() {

	}


	public double backward(double[] y) {
		return 0;
	}


	public double backward(double y) {
		return 0;
	}
	
	
	public Volume[] response() {
		return null;
	}


	public boolean training() {
		return net.inTraining();
	}


	public void net(ConvNet convNet) {
		_net = convNet;
	}


	public ConvNet net() {
		return _net;
	}


	public Layer next() {
		return _next;
	}


	public Layer last() {
		return _last;
	}


	public void next(Layer l) {
		_next = l;
	}


	public void last(Layer l) {		
		l.next(this);					
	}


	public int inW() {
		return sizes[0][0];
	}


	public int inH() {
		return sizes[0][1];
	}


	public int inD() {
		return sizes[0][2];
	}


	public Layer inW(int w) {
		sizes[0][0] = w;
		return this;
	}


	public Layer inH(int h) {
		sizes[0][1] = h;
		return this;
	}


	public Layer inD(int d) {
		sizes[0][2] = d;
		return this;
	}


	public int outW() {
		return sizes[1][0];
	}


	public int outH() {
		return sizes[1][1];
	}


	public int outD() {
		return sizes[1][2];
	}


	public Layer outW(int w) {
		sizes[1][0] = w;
		return this;
	}


	public Layer outH(int h) {
		sizes[1][1] = h;
		return this;
	}


	public Layer outD(int d) {
		sizes[1][2] = d;
		return this;
	}


	public int inLength() {
		return this.inW() * this.inH() * this.inD();
	}


	public int outLength() {
		return this.outW() * this.outH() * this.outD();
	}	

}