#include <iostream>
#include <fstream>
#include <string>
#include <intrin.h>
#include "Matrix.h"

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



int K = 10;

Matrix Xtrain;
Matrix ytrain;

Matrix Xval;
Matrix yval;

Matrix Xtest;
Matrix ytest;


void loadImages(const char* path, Matrix& X)
{
	std::cout << "Loading images..\n";
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
	
	

	for (int k = 0; k < M; k++)
	{
		for (int i = 0; i < rows * columns; i++)
		{
			int index = k * rows * columns + i;
			
			X(k, i) = (float)(pixels[index]) / 255.f;
		}
	}

	delete[] pixels;
	imageFile.close();
}

void loadLabels(const char* path, Matrix& y)
{
	std::cout << "Loading labels..\n";
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
		y(k) = (float)labels[k];
	}

	delete[] labels;
	labelFile.close();
}



//to training + cross validation
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
	loadImages("Data\\train-images.idx3-ubyte", X);
	
	loadLabels("Data\\train-labels.idx1-ubyte", y);
	
	splitTrainingData(0.75, X, y, Xtrain, ytrain, Xval, yval);

	loadImages("Data\\t10k-images.idx3-ubyte", Xtest);

	loadLabels("Data\\t10k-labels.idx1-ubyte", ytest);
	
}




int main()
{
	

	//loadData();

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

	return 0;
}