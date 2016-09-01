
#include "NeuralNetwork.h"

#include <random>
#include "GradientDescent.h"

NeuralNetwork::NeuralNetwork(int inputSize, float lambda) : inputSize{ inputSize }, lambda{ lambda }
{

}



void NeuralNetwork::addSimpleLayer(int nodeSize)
{
	int w = layers.size() == 0 ? inputSize : layers.back()->Theta.N() + 1;
	layers.push_back(std::make_shared<SimpleHiddenLayer>(nodeSize, w));
	outputSize = nodeSize;
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

void NeuralNetwork::train(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& xval, const Matrix& yval)
{
	LayerStructure ls = getLayerStructure();
	
	Matrix thetas = getUnrolledThetas(ls.thetaCount);
	
	auto cost = [&](const Matrix& theta, Matrix& grad) {
		return NeuralNetwork::costFunction(theta, ls, outputSize, Xtrain, ytrain, lambda, grad);
	};

	gradientDescent(cost, thetas, 1.0f, 100);

	//update thetas
	std::vector<Matrix> reshapedThetas = getReshaped(thetas, ls);
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i]->Theta = reshapedThetas[i];
	}
}


float NeuralNetwork::costFunction(const Matrix& thetasUnrolled, const LayerStructure& layerStructure, int K, const Matrix& X, const Matrix& y, float lambda, Matrix& gradientUnrolled)
{

	std::vector<Matrix> thetas = getReshaped(thetasUnrolled, layerStructure);
	
	int m = X.N();

	//yb: binary labels
	Matrix yb{ y.N(), K };
	for (int i = 0; i < m; i++)
	{
		yb(i, (int)(y(i) + 0.1f)) = 1.0f;
	}

	//Matrix h = hypothesis(X, thetas);

	std::vector<Matrix> act;
	std::vector<Matrix> zed;
	
	Matrix a = X; 
	
	float reg = 0.0f;

	//forward prop
	for (int i = 0; i < thetas.size(); i++)
	{
		const Matrix& Theta = thetas[i];

		Matrix z = mulFirstWithSecondTransposedM(a, Theta);
		if (i != thetas.size() - 1)
		{
			a = appendNextToM(onesM(m, 1), sigmoidM(z));
		}
		else
		{
			a = sigmoidM(z);
		}
		
		zed.push_back(z);
		act.push_back(a);
		
		Matrix bias = rangeM(Theta, 0, 0, 1, Theta.N());
		reg += sumSquaredAllM(Theta) - sumSquaredAllM(bias);
	}
	reg *= lambda * 0.5f / m;

	
	float J = -meanAllM(sumByRowsM(mulElementWiseM(yb, logM(a)) + mulElementWiseM(-yb + 1.0f, logM(-a + 1.0f))));
	J += reg;


	//backprop
	std::vector<Matrix> gradThetas{ (size_t)thetas.size() };
	Matrix delta = a - yb;

	for (int i = thetas.size() - 2; i >= 0; i--)
	{
		const Matrix& Theta = thetas[i + 1];
		Matrix biasClearedTheta = Theta;
		copyMatInM(zeros(biasClearedTheta.N(), 1), biasClearedTheta, 0, 0);
		Matrix cutTheta = rangeM(Theta, 0, 1, Theta.M() - 1, Theta.N());

		gradThetas[i + 1] = (delta.transpose() * act[i]) * (1.0f / (float)m) + biasClearedTheta * (lambda / (float)m);
		delta = mulElementWiseM(delta * cutTheta, sigmoidGradientM(zed[i]));
	}

	{
		Matrix biasClearedTheta0 = thetas[0];
		copyMatInM(zeros(biasClearedTheta0.N(), 1), biasClearedTheta0, 0, 0);
		gradThetas[0] = (delta.transpose() * X) * (1.0f / (float)m) + biasClearedTheta0 * (lambda / (float)m);
	}


	gradientUnrolled = unrollAllM(gradThetas);

	return J;
}

Matrix NeuralNetwork::hypothesis(const Matrix& X, const std::vector<Matrix>& thetas)
{
	int m = X.N();

	//X already has the ones(..)
	Matrix a = sigmoidM(
		mulFirstWithSecondTransposedM(X, thetas[0])
	);

	for (int i = 1; i < thetas.size(); i++)
	{
		a = sigmoidM(
			mulFirstWithSecondTransposedM(
				appendNextToM(onesM(m, 1), a), thetas[i])
		);

	}
	return a;
}


Matrix NeuralNetwork::predict(const Matrix& X)
{
	std::vector<Matrix> thetas;
	thetas.reserve(layers.size());
	for (int i = 0; i < layers.size(); i++)
	{
		thetas.push_back(layers[i]->Theta);
	}
	Matrix h = hypothesis(X, thetas);
	
	return maxIndexByRowsM(h);
}


std::vector<Matrix> NeuralNetwork::getReshaped(const Matrix& unrolled, const LayerStructure& layerStructure)
{
	std::vector<Matrix> thetas;
	int startIndex = 0;
	const std::vector<int>& dim = layerStructure.thetaDimensions;
	for (int i = 0; i < dim.size(); i += 2)
	{
		int n = dim[i];
		int m = dim[i + 1];
		thetas.push_back(reshapeM(unrolled, startIndex, n, m));
		startIndex += n * m;
	}

	return thetas;
}

Matrix NeuralNetwork::getUnrolledThetas(int thetaCount)
{
	if (thetaCount == -1)
	{
		thetaCount = getLayerStructure().thetaCount;
	}

	Matrix thetas{ 1, thetaCount };
	thetas.values.clear();

	for (int i = 0; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;
		thetas.values.insert(thetas.values.end(), Theta.values.cbegin(), Theta.values.cend());
	}
	
	return thetas;
}

LayerStructure NeuralNetwork::getLayerStructure()
{
	std::vector<int> thetaDims((size_t)layers.size() * 2);

	//unrolling thetas
	int thetaCount = 0;

	for (int i = 0; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;
		thetaCount += Theta.N() * Theta.M();
		thetaDims[2 * i] = Theta.N();
		thetaDims[2 * i + 1] = Theta.M();
	}
	return LayerStructure{ thetaDims, thetaCount };
}