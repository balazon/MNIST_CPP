#include <iostream>
#include <fstream>
#include <string>
#include <intrin.h>
#include <numeric>
#include <algorithm>
#include <random>

#include "NeuralNetwork.h"
#include "Matrix.h"
#include "GradientDescent.h"
#include "Timer.h"

//on intel processors this must be defined so that the little endian data gets swapped
#define HIGH_ENDIAN



int swapEndian(int32_t value)
{
	return _byteswap_ulong(value);
}


void readInt(std::ifstream& file, int* dest)
{
	int val = 0;
	file.read((char*)&val, sizeof(val));
#ifdef HIGH_ENDIAN
	val = swapEndian(val);
#endif
	*dest = val;
}

//data variables
Matrix Xtrain;
Matrix ytrain;

Matrix Xval;
Matrix yval;

Matrix Xtest;
Matrix ytest;

void loadImages(const char* path, Matrix& X, std::vector<int>& shuffleIndexes)
{
	std::ifstream imageFile{ path, std::ios::in | std::ios::binary | std::ios::ate };
	if (!imageFile.is_open())
	{
		return;
	}

	imageFile.seekg(0, std::ios::beg);

	int magic = 0;
	int M = 0;
	int rows = 0;
	int columns = 0;

	readInt(imageFile, &magic);
	readInt(imageFile, &M);
	readInt(imageFile, &rows);
	readInt(imageFile, &columns);


	unsigned char* pixels = new unsigned char[M * rows * columns];
	imageFile.read((char*)pixels, M * rows * columns);
	X = Matrix(M, rows * columns);
	
	
	shuffleIndexes.clear();
	shuffleIndexes.reserve(M);
	for (int i = 0; i < M; i++)
	{
		shuffleIndexes.push_back(i);
	}
	unsigned int seed = (unsigned int)(std::chrono::system_clock::now().time_since_epoch().count());
	std::shuffle(shuffleIndexes.begin(), shuffleIndexes.end(), std::default_random_engine(seed));

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < rows * columns; i++)
		{
			int index = k * rows * columns + i;
			
			X(shuffleIndexes[k], i) = (float)(pixels[index]) / 255.f;
		}
	}

	delete[] pixels;
	imageFile.close();
}

void loadLabels(const char* path, Matrix& y, std::vector<int>& shuffleIndexes)
{
	std::ifstream labelFile{ path, std::ios::in | std::ios::binary | std::ios::ate };
	if (!labelFile.is_open())
	{
		return;
	}

	labelFile.seekg(0, std::ios::beg);

	int magic = 0;
	int M = 0;

	readInt(labelFile, &magic);
	readInt(labelFile, &M);

	unsigned char* labels = new unsigned char[M];
	labelFile.read((char*)labels, M);
	y = Matrix(M, 1);

	for (int k = 0; k < M; k++)
	{
		y(shuffleIndexes[k]) = (float)labels[k];
	}

	delete[] labels;
	labelFile.close();
}



//splits to training + cross validation
void splitTrainingData(float trainRatio, const Matrix& X, const Matrix& y, Matrix& Xtrain, Matrix& ytrain, Matrix& Xval, Matrix& yval)
{
	int m = X.N();
	int numTrain = (int)(m * trainRatio);
	int numVal = m - numTrain;

	int w = X.M();

	Xtrain = rangeM(X, 0, 0, w, numTrain);
	ytrain = rangeM(y, 0, 0, 1, numTrain);

	Xval = rangeM(X, numTrain, 0, w, numVal);
	yval = rangeM(y, numTrain, 0, 1, numVal);
}


void loadData()
{
	Matrix X;
	Matrix y;

	//to shuffle data that is read, the same shuffle must be used for labels
	std::vector<int> shuffleIndexes;

	printf("  Loading train images..\n");
	loadImages("Data\\train-images.idx3-ubyte", X, shuffleIndexes);
	
	printf("  Loading train labels..\n");
	loadLabels("Data\\train-labels.idx1-ubyte", y, shuffleIndexes);
	
	printf("  Splitting train data..\n");
	splitTrainingData(0.75, X, y, Xtrain, ytrain, Xval, yval);

	
	printf("  Loading test images..\n");
	loadImages("Data\\t10k-images.idx3-ubyte", Xtest, shuffleIndexes);
	
	printf("  Loading test labels..\n");
	loadLabels("Data\\t10k-labels.idx1-ubyte", ytest, shuffleIndexes);
}

void normalizeData()
{
	float mean = 0.0f;
	float stdev = 0.0f;
	Xtrain = normalizeM(Xtrain, mean, stdev);
	

	Xval = Xval - mean;
	Xval = Xval / stdev;
	

	Xtest = Xtest - mean;
	Xtest = Xtest / stdev;
}

void addBiasToData()
{
	Xtrain = appendNextToM(onesM(Xtrain.N(), 1), Xtrain);

	Xval = appendNextToM(onesM(Xval.N(), 1), Xval);

	Xtest = appendNextToM(onesM(Xtest.N(), 1), Xtest);
}

void testGradientDescent()
{
	// f(x) = (x - 4)^2, x0 = 5.5
	// f'(x) = 2x - 8
	std::vector<float> x = { 5.5f };
	auto cost = [](const std::vector<float>& theta, std::vector<float>& grad) {
		grad[0] = 2.f * theta[0] - 8.f;
		return (theta[0] - 4.f) * (theta[0] - 4.f);
	};
	gradientDescent(cost, x, 0.1f, 100);

	//result should be close to 4.0f
	printf("gradient descent result: %.2f", x[0]);
}

void testMath()
{
	Matrix a(3, 3);
	a(0, 0) = 1;
	a(1, 1) = 2;
	a(1, 2) = 6;
	a(2, 0) = 4;
	a(2, 2) = 3;

	Matrix b(3, 3);
	b(0, 0) = 5;
	b(0, 1) = 4;
	b(0, 2) = 3;
	b(1, 0) = 1;
	b(1, 1) = 9;
	b(1, 2) = 7;
	b(2, 0) = 2;
	b(2, 1) = 9;
	b(2, 2) = -2;

	Matrix c = a * b;

	printMx(c);

	printMx(-c);

	Matrix d(4, 3);
	d(0, 0) = 5;
	d(0, 1) = 4;
	d(0, 2) = 3;
	d(1, 0) = 1;
	d(1, 1) = 9;
	d(1, 2) = 7;
	d(2, 0) = 2;
	d(2, 1) = 9;
	d(2, 2) = -2;
	d(3, 0) = 5;
	d(3, 1) = -1;
	d(3, 2) = 4;

	std::cout << "\n";
	printMx(d);
	std::cout << "\n";
	printMx(d.transpose());

	std::cout << "\n";
	Matrix e = appendBelowM(onesM(1, 3), d);
	printMx(e);

	std::cout << "\n";
	Matrix f = appendNextToM(onesM(4, 1), d);
	printMx(f);

	std::cout << "\n";
	printMx(maxIndexByRowsM(f));
	std::cout << "\n";

	Matrix g{ 10, 1 };
	Matrix h{ 10, 1 };
	g(0) = 2;
	g(1) = 1;
	g(2) = 5;
	g(3) = 2;
	g(4) = 7;
	g(5) = 3;
	g(6) = 7;
	g(7) = 4;
	g(8) = 4;
	g(9) = 1;

	h(0) = 1;
	h(1) = 5;
	h(2) = 5;
	h(3) = 2;
	h(4) = 3;
	h(5) = 3;
	h(6) = 3;
	h(7) = 4;
	h(8) = 1;
	h(9) = 1;
	printMx(g);
	std::cout << "\n";
	printMx(h);
	std::cout << "\n";
	printMx(g == h);
	std::cout << "\n" << meanAllM(g == h) << "\n";

	
	Matrix stats{ 1, 5 };
	stats(0) = 2.0f;
	stats(1) = 0.0f;
	stats(2) = 4.0f;
	stats(3) = 4.0f;
	stats(4) = 5.0f;

	printMx(stats);
	std::cout << "\n";
	printf("mean: %.2f\nstdev: %.2f\n", meanAllM(stats), standardDevM(stats));
}

void testMxMul()
{
	//multiplying big randomized matrices

	std::default_random_engine generator;

	std::uniform_real_distribution<float> distribution{ -100.0f, 100.0f };

	Matrix A{ 1000, 1000 };
	Matrix B{ 1000, 1000 };

	for (int i = 0; i < A.N() * A.M(); i++)
	{
		A(i) = distribution(generator);
	}
	for (int i = 0; i < B.N() * B.M(); i++)
	{
		B(i) = distribution(generator);
	}


	printf("multest A * B\n");

	Timer::Instance().start();
	
	Matrix C = A * B;

	printf("Took: %llu millis\n", Timer::Instance().endMillisElapsed());
}

void neural()
{
	printf("Loading data..\n");
	loadData();
	printf("Loading data finished.\n");

	printf("Normalizing data\n");
	normalizeData();

	printf("Adding bias\n");
	addBiasToData();

	printf("Creating NN\n");
	NeuralNetwork nn{ Xtrain.M() };

	
	// to use tanh as activation, use addSimpleLayer(nodeSize, tanhM, tanhGradientM)
	nn.addSimpleLayer(200);

	nn.addSimpleLayer(50);

	//last layer is the output layer
	// - don't use tanhM as activation on the last layer
	// (tanh may evaluate to negative numbers, cost function will use log on them : NaN)
	nn.addSimpleLayer(10);


	printf("NN train \n");
	std::vector<float> lambdas = { 0.0f};
	int epochs = 1;
	int batchSize = 1000;
	float learningRate = 1.0f;
	int gradIter = 100;

	nn.trainComplete(Xtrain, ytrain, Xval, yval, epochs, batchSize, lambdas, learningRate, gradIter);

	


	printf("NN predict\n");
	Matrix p = nn.predict(Xtest);

	printf("Test accuracy: %f\n", meanAllM(p == ytest));
}

int main()
{
	//testGradientDescent();

	//testMath();

	//testMxMul();
	
	neural();

	return 0;
}