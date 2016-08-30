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
	NeuralNetwork(int inputSize, float lambda) : inputSize{ inputSize }, lambda{ lambda }
	{

	}

	void addSimpleLayer(int nodeSize);
	

	void addOutputLayer(int outputSize);
	

	void train(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& xval, const Matrix& yval) {}

	void validate() {}
};

class SimpleHiddenLayer
{

public:

	Matrix Theta;

	SimpleHiddenLayer(int n, int m) : Theta(n, m)
	{

	}
};