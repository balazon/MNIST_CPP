
#include "NeuralNetwork.h"

#include <random>

NeuralNetwork::NeuralNetwork(int inputSize, float lambda) : inputSize{ inputSize }, lambda{ lambda }
{

}



void NeuralNetwork::addSimpleLayer(int nodeSize)
{
	int w = layers.size() == 0 ? inputSize : layers.back()->Theta.N() + 1;
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


float NeuralNetwork::costFunction(const Matrix& X, const Matrix& y, int K, Matrix& gradient)
{
	int m = X.N();

	//yb: binary labels
	Matrix yb{ y.N(), K };
	for (int i = 0; i < m; i++)
	{
		yb(i, (int)(y(i) + 0.1f)) = 1.0f;
	}

	Matrix h = hypothesis(X);
	
	float reg = 0.0f;
	for (int i = 0; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;
		Matrix bias = rangeM(Theta, 0, 0, 1, Theta.N());
		reg += sumSquaredAllM(Theta) - sumSquaredAllM(bias);
	}
	reg *= lambda * 0.5f / m;

	
	float J = -meanAllM(sumByRowsM(mulElementWiseM(yb, logM(h)) + mulElementWiseM(-yb + 1.0f, logM(-h + 1.0f))));
	J += reg;

	return J;
}

Matrix NeuralNetwork::hypothesis(const Matrix& X)
{
	int m = X.N();

	//X already has the ones(..)
	Matrix a = sigmoidM(
		mulFirstWithSecondTransposedM(X, layers[0]->Theta)
	);

	for (int i = 1; i < layers.size(); i++)
	{
		a = sigmoidM(
			mulFirstWithSecondTransposedM(
				appendNextToM(onesM(m, 1), a), layers[i]->Theta)
		);

	}
	return a;
}


Matrix NeuralNetwork::predict(const Matrix& X)
{
	Matrix h = hypothesis(X);
	
	return maxIndexByRowsM(h);
}