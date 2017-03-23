import ml.convnet.ConvNet;
import ml.convnet.layer.*;
import ml.convnet.layer.activation.*;
import ml.convnet.trainer.*;


public class Test {
	public static void main(String[] args) {

		ConvNet net = new ConvNet();

		net.addLayer(new Input(1, 1, 2))
				.addLayer(new FullConnect(10, 1.0))
				.addLayer(new LeRu())
				.addLayer(new FullConnect(1, 1.0))
				.addLayer(new Sigmoid());
		
		net.trainer(new SGDTrainer(0.01, 1, 0, 0, 0));
		

		// create layer of 10 linear neurons (no activation function by default)
		// {type:'fc', num_neurons:10}
		// // create layer of 10 neurons that use sigmoid activation function
		// {type:'fc', num_neurons:10, activation:'sigmoid'} // x->1/(1+e^(-x))
		// {type:'fc', num_neurons:10, activation:'tanh'} // x->tanh(x)
		// {type:'fc', num_neurons:10, activation:'relu'} // rectified linear units: x->max(0,x)
		// // maxout units: (x,y)->max(x,y). num_neurons must be divisible by 2.
		// // maxout "consumes" multiple filters for every output. Thus, this line
		// // will actually produce only 5 outputs in this layer. (group_size is 2)
		// // by default.
		// {type:'fc', num_neurons:10, activation:'maxout'}
		// // specify group size in maxout. num_neurons must be divisible by group_size.
		// // here, output will be 3 neurons only (3 = 12/4)
		// {type:'fc', num_neurons:12, group_size: 4, activation:'maxout'}
		// // dropout half the units (probability 0.5) in this layer during training, for regularization
		// {type:'fc', num_neurons:10, activation:'relu', drop_prob: 0.5}

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
