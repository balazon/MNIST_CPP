
#include "NeuralNetwork.h"

#include <random>


void NeuralNetwork::addSimpleLayer(int nodeSize)
{
	int w = layers.size() == 0 ? inputSize + 1 : layers.back()->Theta.N() + 1;
	layers.push_back(std::make_shared<SimpleHiddenLayer>(nodeSize, w));
}

void NeuralNetwork::initWeights()
{
	std::default_random_engine generator;

	for (int i = 0; i < layers.size(); i++)
	{
		Matrix& Theta = layers[i]->Theta;
		int size = Theta.N() * Theta.M();
		float eps = 2.45f / sqrtf((float)size);

		std::uniform_real_distribution<float> distribution{ -eps, eps };
		for (int j = 0; j < size; j++)
		{
			Theta(j) = distribution(generator);
		}
	}
}


Matrix NeuralNetwork::predict(const Matrix& X)
{
	/*m = size(X, 1);
	num_labels = size(Theta2, 1);

	% You need to return the following variables correctly
		p = zeros(size(X, 1), 1);

	h1 = sigmoid([ones(m, 1) X] * Theta1');
		h2 = sigmoid([ones(m, 1) h1] * Theta2');
			[dummy, p] = max(h2, [], 2);*/


	/*
	a1 = [ones(m, 1) X];
	a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
	a3 = sigmoid(a2 * Theta2');
	[~, p] = max(a3, [], 2);

	*/

	int m = X.N();
	
	Matrix a = X;
	for (int i = 0; i < layers.size(); i++)
	{
		a = sigmoidM(appendNextToM(onesM(m, 1), a) * layers[i]->Theta.transpose());
	}
	return maxIndexByRowsM(a);
}