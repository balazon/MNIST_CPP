#pragma once

#include <vector>
#include <memory>
#include <functional>
#include "Matrix.h"

class SimpleHiddenLayer;

struct LayerStructure
{
	std::vector<int> thetaDimensions;
	int thetaCount;

	std::vector<std::function<Matrix(const Matrix&)>> actFuns;
	std::vector<std::function<Matrix(const Matrix&)>> actFunGrads;

	LayerStructure() {}

	LayerStructure(const std::vector<int>& thetaDimensions, int thetaCount,
		std::vector<std::function<Matrix(const Matrix&)>> actFuns,
		std::vector<std::function<Matrix(const Matrix&)>> actFunGrads)
		: thetaDimensions{ thetaDimensions }, thetaCount{ thetaCount }, actFuns{ actFuns }, actFunGrads{ actFunGrads }
	{

	}
};

class NeuralNetwork
{
	int inputSize;

	std::vector<std::shared_ptr<SimpleHiddenLayer>> layers;

	int outputSize;

	bool firstBatchGrad;
	int allBatchCount;

	LayerStructure currentLayerStructure;


	void updateLayerStructure();

public:
	NeuralNetwork(int inputSize);

	void addSimpleLayer(int nodeSize,
	                    std::function<Matrix(const Matrix&)> actFun = sigmoidM,
	                    std::function<Matrix(const Matrix&)> actFunGrad = sigmoidGradientM);

	void initWeights();
	

	void trainComplete(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval,
	                   int epochs, int batchSize, std::vector<float> lambdas, float alpha, int optIter);

	void trainWithLambda(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval,
	                     int epochs, int batchSize, float lambda, float alpha, int optIter);

	void trainStep(const Matrix& Xtrain, const Matrix& ytrain, float lambda, float alpha, int optIter);

	static float costFunction(const Matrix& thetasUnrolled, const LayerStructure& layerStructure, int K,
	                          const Matrix& X, const Matrix& y, float lambda, Matrix& gradientUnrolled);

	Matrix hypothesis(const Matrix& X);

	Matrix predict(const Matrix& X);

	static std::vector<Matrix> getReshaped(const Matrix& unrolled, const LayerStructure& layerStructure);

	Matrix getUnrolledThetas();

	void setThetas(const std::vector<Matrix>& unrolled);

	void saveThetas(std::ostream& stream);

	void saveFirstLayerVisualization(std::ostream& stream, int rows, int cols, int miniRows, int miniCols);
};

class SimpleHiddenLayer
{

public:

	Matrix Theta;

	std::function<Matrix(const Matrix&)> actFun;
	std::function<Matrix(const Matrix&)> actFunGrad;

	SimpleHiddenLayer(int n, int m,
	                  std::function<Matrix(const Matrix&)> actFun,
	                  std::function<Matrix(const Matrix&)> actFunGrad)
		: Theta(n, m), actFun{ actFun }, actFunGrad{ actFunGrad }
	{

	}
};


