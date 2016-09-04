#pragma once

#include <vector>
#include <iostream>

class Matrix
{
	
	int n, m;

public:
	std::vector<float> values;

	Matrix();
	Matrix(int n, int m, float defVal = 0.0f);

	Matrix(
		int n, int m,
		std::vector<float>::const_iterator first,
		std::vector<float>::const_iterator last,
		const std::vector<float>::allocator_type& alloc = std::vector<float>::allocator_type());

	Matrix& operator=(Matrix other);

	Matrix(const Matrix& other);

	Matrix(Matrix&& other);

	float& operator()(int i);

	float operator()(int i) const;

	float& operator()(int i, int j);

	float operator()(int i, int j) const;

	int N() const;

	int M() const;

	Matrix transpose() const;

	
	friend void swap(Matrix& m1, Matrix& m2);
	
};

void swap(Matrix& m1, Matrix& m2);



std::ostream& operator<<(std::ostream& stream, const Matrix& m);

void printMx(const Matrix& m);

void visualizeLayerM(std::ostream& stream, const Matrix& mat, int rows, int cols, int miniRows, int miniCols);


Matrix operator-(const Matrix& m);

Matrix operator+(const Matrix& m1, const Matrix& m2);

Matrix operator-(const Matrix& m1, const Matrix& m2);

Matrix mulFirstWithSecondTransposedM(const Matrix& m1, const Matrix& m2);

Matrix operator*(const Matrix& m1, const Matrix& m2);

Matrix operator==(const Matrix& m1, const Matrix& m2);

Matrix mulElementWiseM(const Matrix& m1, const Matrix& m2);


Matrix operator+(const Matrix& m, float val);

Matrix operator-(const Matrix& m, float val);

Matrix operator*(const Matrix& m, float val);

Matrix operator/(const Matrix& m, float val);


Matrix sigmoidM(const Matrix& m);

Matrix sigmoidGradientM(const Matrix& m);

Matrix tanhM(const Matrix& m);

Matrix tanhGradientM(const Matrix& m);

Matrix logM(const Matrix& m);

//cuts out a part of a Matrix into a new Matrix
Matrix rangeM(const Matrix& m, int i, int j, int w, int h);

Matrix appendBelowM(const Matrix& m1, const Matrix& m2);

void copyMatInM(const Matrix& source, Matrix& dest, int i, int j);

Matrix appendNextToM(const Matrix& m1, const Matrix& m2);

Matrix reshapeM(const Matrix& thetas, int startIndex, int n, int m);

Matrix unrollAllM(const std::vector<Matrix>& matrices);

Matrix zeros(int i, int j);

Matrix onesM(int i, int j);

Matrix maxIndexByRowsM(const Matrix& m);

Matrix sumByRowsM(const Matrix& m);

float sumAllM(const Matrix& m);

float meanAllM(const Matrix& m);

float sumSquaredAllM(const Matrix& m);

float standardDevM(const Matrix& m, float mean);

float standardDevM(const Matrix& m);

Matrix normalizeM(const Matrix& m, float& mean, float& stdev);