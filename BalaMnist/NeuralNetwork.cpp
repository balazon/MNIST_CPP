
#include "NeuralNetwork.h"

#include <random>
#include "GradientDescent.h"
#include "Timer.h"

NeuralNetwork::NeuralNetwork(int inputSize) : inputSize{ inputSize }
{
	
}



void NeuralNetwork::addSimpleLayer(int nodeSize,
	std::function<Matrix(const Matrix&)> actFun,
	std::function<Matrix(const Matrix&)> actFunGrad)
{
	int w = layers.size() == 0 ? inputSize : layers.back()->Theta.N() + 1;
	layers.push_back(std::make_shared<SimpleHiddenLayer>(nodeSize, w, actFun, actFunGrad));
	outputSize = nodeSize;

	updateLayerStructure();
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


void NeuralNetwork::trainComplete(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval,
	int epochs, int batchSize, std::vector<float> lambdas, float alpha, int optIter)
{
	firstBatchGrad = true;
	allBatchCount = (int) lambdas.size() * epochs * (Xtrain.N() + batchSize - 1) / batchSize;

	float minValCost = FLT_MAX;
	float bestLambda = 0.0f;
	Matrix bestThetas;


	std::vector<float> trainCosts;
	std::vector<float> crossCosts;


	for (float lam : lambdas)
	{

		trainWithLambda(Xtrain, ytrain, Xval, yval, epochs, batchSize, lam, alpha, optIter);

		printf("Training with lambda = %.4f complete\n", lam);

		
		Matrix thetasUnrolled = getUnrolledThetas();
		Matrix grad;
		float trainJ = costFunction(thetasUnrolled, currentLayerStructure, outputSize, Xtrain, ytrain, 0.0f, grad);
		float crossJ = costFunction(thetasUnrolled, currentLayerStructure, outputSize, Xval, yval, 0.0f, grad);

		printf("Lambda = %.4f cross validation error cost : %.2f\n", lam, crossJ);

		trainCosts.push_back(trainJ);
		crossCosts.push_back(crossJ);

		if (minValCost > crossJ)
		{
			minValCost = crossJ;
			bestLambda = lam;
			bestThetas = thetasUnrolled;
		}
	}


	std::vector<Matrix> reshapedThetas = getReshaped(bestThetas, currentLayerStructure);
	setThetas(reshapedThetas);

	printf("\nCross validation results: \n");
	for (int i = 0; i < lambdas.size(); i++)
	{
		printf("  Lambda = %.4f train cost: %.4f, cross cost: %.4f\n", lambdas[i], trainCosts[i], crossCosts[i]);
	}

	
}

void NeuralNetwork::trainWithLambda(const Matrix& Xtrain, const Matrix& ytrain, const Matrix& Xval, const Matrix& yval,
	int epochs, int batchSize, float lambda, float alpha, int optIter)
{
	initWeights();

	for (int i = 0; i < epochs; i++)
	{
		printf("Epoch %d start\n", (i + 1));

		int m = Xtrain.N();

		int batchIterations = m / batchSize;
		if (m % batchSize != 0)
		{
			batchIterations++;
		}

		int currentPosition = 0;
		for (int j = 0; j < batchIterations; j++)
		{
			printf("Epoch %d batch %d \n", (i + 1), (j + 1));
			int currentBatchSize = (currentPosition + batchSize <= m) ? batchSize : (m - currentPosition);

			Matrix X = rangeM(Xtrain, currentPosition, 0, Xtrain.M(), currentBatchSize);
			Matrix y = rangeM(ytrain, currentPosition, 0, ytrain.M(), currentBatchSize);

			
			if (firstBatchGrad)
			{
				Timer::Instance().start();
			}

			trainStep(X, y, lambda, alpha, optIter);

			if (firstBatchGrad)
			{
				unsigned long long seconds = ((allBatchCount - 1) * Timer::Instance().endMillisElapsed()) / 1000;
				int minutes = (int)(seconds / 60);
				seconds -= minutes * 60;

				printf("\n---\nEstimated finish after %d minutes %llu seconds\n---\n", minutes, seconds);

				firstBatchGrad = false;
			}
			

			currentPosition += currentBatchSize;
		}

	}
}



void NeuralNetwork::trainStep(const Matrix& Xtrain, const Matrix& ytrain, float lambda, float alpha, int optIter)
{
	Matrix thetas = getUnrolledThetas();
	
	auto cost = [&](const Matrix& theta, Matrix& grad) {
		return NeuralNetwork::costFunction(theta, currentLayerStructure, outputSize, Xtrain, ytrain, lambda, grad);
	};

	gradientDescent(cost, thetas, alpha, optIter);

	//update thetas
	std::vector<Matrix> reshapedThetas = getReshaped(thetas, currentLayerStructure);
	setThetas(reshapedThetas);
}


float NeuralNetwork::costFunction(const Matrix& thetasUnrolled, const LayerStructure& layerStructure, int K,
	const Matrix& X, const Matrix& y, float lambda, Matrix& gradientUnrolled)
{

	std::vector<Matrix> thetas = getReshaped(thetasUnrolled, layerStructure);
	const std::vector<std::function<Matrix(const Matrix&)>>& actFuns = layerStructure.actFuns;
	const std::vector<std::function<Matrix(const Matrix&)>>& actFunGrads = layerStructure.actFunGrads;

	int m = X.N();

	//yb: binary labels
	Matrix yb{ y.N(), K };
	for (int i = 0; i < m; i++)
	{
		yb(i, (int)(y(i) + 0.1f)) = 1.0f;
	}


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
			a = appendNextToM(onesM(m, 1), actFuns[i](z));
		}
		else
		{
			a = actFuns[i](z);
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

	for (int i = (int)thetas.size() - 2; i >= 0; i--)
	{
		const Matrix& Theta = thetas[i + 1];
		Matrix biasClearedTheta = Theta;
		copyMatInM(zeros(biasClearedTheta.N(), 1), biasClearedTheta, 0, 0);
		Matrix cutTheta = rangeM(Theta, 0, 1, Theta.M() - 1, Theta.N());

		gradThetas[i + 1] = (delta.transpose() * act[i]) * (1.0f / (float)m) + biasClearedTheta * (lambda / (float)m);
		delta = mulElementWiseM(delta * cutTheta, actFunGrads[i](zed[i]));
	}

	{
		Matrix biasClearedTheta0 = thetas[0];
		copyMatInM(zeros(biasClearedTheta0.N(), 1), biasClearedTheta0, 0, 0);
		gradThetas[0] = (delta.transpose() * X) * (1.0f / (float)m) + biasClearedTheta0 * (lambda / (float)m);
	}


	gradientUnrolled = unrollAllM(gradThetas);

	return J;
}

Matrix NeuralNetwork::hypothesis(const Matrix& X)
{
	int m = X.N();

	//X already has the ones(..)
	
	Matrix a = layers[0]->actFun(
		mulFirstWithSecondTransposedM(X, layers[0]->Theta)
	);

	for (int i = 1; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;

		a = layers[i]->actFun(
			mulFirstWithSecondTransposedM(
				appendNextToM(onesM(m, 1), a), Theta)
		);

	}
	return a;
}


Matrix NeuralNetwork::predict(const Matrix& X)
{
	
	Matrix h = hypothesis(X);
	
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

Matrix NeuralNetwork::getUnrolledThetas()
{
	Matrix thetas{ 1, currentLayerStructure.thetaCount };
	thetas.values.clear();

	for (int i = 0; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;
		thetas.values.insert(thetas.values.end(), Theta.values.cbegin(), Theta.values.cend());
	}
	
	return thetas;
}

void NeuralNetwork::updateLayerStructure()
{
	std::vector<int> thetaDims((size_t)layers.size() * 2);

	//unrolling thetas
	int thetaCount = 0;

	std::vector<std::function<Matrix(const Matrix&)>> actFuns;
	std::vector<std::function<Matrix(const Matrix&)>> actFunGrads;

	for (int i = 0; i < layers.size(); i++)
	{
		const Matrix& Theta = layers[i]->Theta;
		thetaCount += Theta.N() * Theta.M();
		thetaDims[2 * i] = Theta.N();
		thetaDims[2 * i + 1] = Theta.M();

		actFuns.push_back(layers[i]->actFun);
		actFunGrads.push_back(layers[i]->actFunGrad);
	}
	currentLayerStructure = LayerStructure{ thetaDims, thetaCount, actFuns, actFunGrads};
}

void NeuralNetwork::setThetas(const std::vector<Matrix>& unrolled)
{
	for (int i = 0; i < layers.size(); i++)
	{
		layers[i]->Theta = unrolled[i];
	}
}

void NeuralNetwork::saveThetas(std::ostream& stream)
{
	for (int i = 0; i < layers.size(); i++)
	{
		stream << layers[i]->Theta;
		stream << std::endl;
	}
}

void NeuralNetwork::saveFirstLayerVisualization(std::ostream& stream, int rows, int cols, int miniRows, int miniCols)
{
	visualizeLayerM(stream, layers[0]->Theta, rows, cols, miniRows, miniCols);
}