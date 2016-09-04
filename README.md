# MNIST_CPP

This is my attempt for creating a neural network that learns the MNIST data.

The data can be found on this link:

http://yann.lecun.com/exdb/mnist/

(it can also be found in this repository in the BalaMnist/Data folder)

Usage for the MNIST data is the following:

	printf("Loading data..\n");

	// You may need to customize this function and the other functions called from here
	// if you intend to use it for some other data
	loadData();

	printf("Loading data finished.\n");


	printf("Normalizing data\n");

	normalizeData();


	printf("Adding bias\n");

	// This is important, the NN assumes the data already has the bias as the first column
	addBiasToData();


	printf("Creating NN\n");

	NeuralNetwork nn{ Xtrain.M() };



	// Set custom parameters here

	// Add a layer using addSimpleLayer function with the number of the nodes specified
	// To use tanh as activation, use addSimpleLayer(nodeSize, tanhM, tanhGradientM)
	nn.addSimpleLayer(200);

	nn.addSimpleLayer(50);

	// Last layer is the output layer - number of nodes should equal the number of labels in the classification data
	// - don't use tanhM as activation on the last layer
	// (tanh may evaluate to negative numbers, cost function will use log on them : NaN)
	nn.addSimpleLayer(10);


	printf("NN train \n");

	// The list of lambdas (regularization parameters) to try, after training the neural network with each lambda
	// the NN will choose the one with the least amount of cross validation error
	std::vector<float> lambdas = { 0.0f, 0.01f, 0.03f, 0.1f, 0.3f, 1.0f, 3.0f, 10.0f };
	int epochs = 1;
	int batchSize = 1000;
	float learningRate = 0.3f;
	int gradIter = 200;

	nn.trainComplete(Xtrain, ytrain, Xval, yval, epochs, batchSize, lambdas, learningRate, gradIter);


	std::ofstream myfile;
	myfile.open("weights.txt");

	// To save layer weights, you can call saveThetas function with an ofstream file object as parameter
	// The weights will be written out as matrices separated by comma and space, row by row
	nn.saveThetas(myfile);
	myfile.close();

	
	std::ofstream file;
	file.open("firstlayer.txt");

	// To save the first layer in an n rows by m columns rearrangement with each neuron's weights as a 2d picture
	// that can be copied in excel to visualize, use the function saveFirstLayerVisualization
	nn.saveFirstLayerVisualization(file, 10, 20, 28, 28);
	file.close();


	printf("NN predict\n");

	// p contains the neural network's predicted labels
	Matrix p = nn.predict(Xtest);

	// Test accuracy is achieved by comparing the predicted labels to the test labels
	// which is a column matrix with 0's and 1's in it (1 when the prediction matched the test)
	// and then you take the average from those values using meanAllM
	printf("Test accuracy: %f\n", meanAllM(p == ytest));

Running this code yielded a 96,9 % test score with the regularization parameter lambda = 0.03

I also visualized the first layer (200 pictures in a 10 by 20 arrangement):
![alt tag](https://cloud.githubusercontent.com/assets/3685997/18227874/36f2b448-7234-11e6-9a0a-5d312cfe529e.png)

Usage for other data should be similar, but this project is not intended to be used as a deep learning library, since it has no such optimizations/parallelizations as the big frameworks out there (caffee, theano, tensorflow, etc), and as such it will run much slower than those mentioned frameworks. However it is a nice learning project for neural networks.

This project is based on the material in the Stanford coursera Machine Learning course: https://www.coursera.org/learn/machine-learning

I would like to express my thanks to my online teacher Andrew Ng who made all this possible.
