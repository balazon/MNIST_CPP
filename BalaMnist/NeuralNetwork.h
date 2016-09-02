#pragma once

#include <vector>
#include <memory>
#include "Matrix.h"

class SimpleHiddenLayer;
struct LayerStructure;

class NeuralNetwork
{
	int inputSize;

	float regLambda;

	std::vector<std::shared_ptr<SimpleHiddenLayer>> layers;

	int outputSize;

public:
	NeuralNetwork(int inputSize);

	void addSimpleLayer(int nodeSize);
	

	void initWeights();
	

	void trainComplete(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval, int epochs, int batchSize, std::vector<float> lambdas, float alpha, int optIter);

	void trainWithLambda(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval, int epochs, int batchSize, float lambda, float alpha, int optIter);

	void trainStep(const Matrix& Xtrain, const Matrix& ytrain, float lambda, float alpha, int optIter);

	void validate() {}

	static float costFunction(const Matrix& thetasUnrolled, const LayerStructure& layerStructure, int K, const Matrix& X, const Matrix& y, float lambda, Matrix& gradientUnrolled);

	static Matrix hypothesis(const Matrix& X, const std::vector<Matrix>& thetas);

	Matrix predict(const Matrix& X);

	static std::vector<Matrix> getReshaped(const Matrix& unrolled, const LayerStructure& layerStructure);

	Matrix getUnrolledThetas(int thetaCount = -1);

	LayerStructure getLayerStructure();

	void setThetas(const std::vector<Matrix>& unrolled);
};

class SimpleHiddenLayer
{

public:

	Matrix Theta;

	SimpleHiddenLayer(int n, int m) : Theta(n, m)
	{

	}
};


struct LayerStructure
{
	std::vector<int> thetaDimensions;
	int thetaCount;

	LayerStructure(const std::vector<int>& thetaDimensions, int thetaCount) : thetaDimensions{ thetaDimensions }, thetaCount{ thetaCount }
	{

	}
};