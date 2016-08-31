#pragma once

#include <vector>
#include <memory>
#include "Matrix.h"

class SimpleHiddenLayer;

class NeuralNetwork
{
	int inputSize;

	float lambda;

	std::vector<std::shared_ptr<SimpleHiddenLayer>> layers;
public:
	NeuralNetwork(int inputSize, float lambda);

	void addSimpleLayer(int nodeSize);
	

	void initWeights();
	

	void train(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& xval, const Matrix& yval) {}

	void validate() {}

	float costFunction(const Matrix& X, const Matrix& y, int K);

	Matrix hypothesis(const Matrix& X);

	Matrix predict(const Matrix& X);
};

class SimpleHiddenLayer
{

public:

	Matrix Theta;

	SimpleHiddenLayer(int n, int m) : Theta(n, m)
	{

	}
};