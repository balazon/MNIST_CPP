#pragma once

#include "Matrix.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <string>



Matrix::Matrix() : Matrix(1, 1) {}
Matrix::Matrix(int n, int m, float defVal) : n{ n }, m{ m }, values((size_t)(n * m), defVal)
{

}

Matrix& Matrix::operator=(Matrix other)
{
	swap(*this, other);
	return *this;
}

Matrix::Matrix(const Matrix& other)
{
	values = other.values;
	n = other.n;
	m = other.m;
}

Matrix::Matrix(Matrix&& other) : Matrix()
{
	swap(*this, other);
}

float& Matrix::operator()(int i)
{
	if (i >= n * m || i < 0)
	{
		std::cout << "out of bounds  at Mx (" << i << ")\n";
	}
	return values[i];
}

float Matrix::operator()(int i) const
{
	if (i >= n * m || i < 0)
	{
		std::cout << "out of bounds  at Mx (" << i << ")\n";
	}
	return values[i];
}

float& Matrix::operator()(int i, int j)
{
	if (i >= n || j >= m || i < 0 || j < 0)
	{
		std::cout << "out of bounds  at Mx (" << i << ", " << j << ")\n";
	}
	return values[i * m + j];
}

float Matrix::operator()(int i, int j) const
{
	if (i >= n || j >= m || i < 0 || j < 0)
	{
		std::cout << "out of bounds  at Mx (" << i << ", " << j << ")\n";
	}
	return values[i * m + j];
}

int Matrix::N() const
{
	return n;
}

int Matrix::M() const
{
	return m;
}

Matrix Matrix::transpose() const
{
	Matrix res{ m, n };
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			res(i, j) = this->operator()(j, i);
		}
	}
	return res;
}


	//friend void swap(Matrix& m1, Matrix& m2);




void swap(Matrix& m1, Matrix& m2)
{
	using std::swap;

	std::swap(m1.values, m2.values);
	std::swap(m1.n, m2.n);
	std::swap(m1.m, m2.m);
}


Matrix operator-(const Matrix& m)
{
	Matrix res = m;

	for (int i = 0; i < m.N() * m.M(); i++)
	{
		res(i) = -res(i);
	}
	return res;
}


Matrix operator+(const Matrix& m1, const Matrix& m2)
{
	if (m1.N() != m2.N() || m1.M() != m2.M())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}


	Matrix res{ m1.N(), m1.M() };
	for (int i = 0; i < m1.N(); i++)
	{
		for (int j = 0; j < m1.M(); j++)
		{
			res(i, j) = m1(i, j) + m2(i, j);
		}
	}

	return res;
}

Matrix operator-(const Matrix& m1, const Matrix& m2)
{
	if (m1.N() != m2.N() || m1.M() != m2.M())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}

	Matrix res{ m1.N(), m1.M() };
	for (int i = 0; i < m1.N(); i++)
	{
		for (int j = 0; j < m1.M(); j++)
		{
			res(i, j) = m1(i, j) - m2(i, j);
		}
	}

	return res;
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	if (m1.M() != m2.N())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}

	Matrix res{ m1.N(), m2.M() };
	for (int i = 0; i < res.N(); i++)
	{
		for (int j = 0; j < res.M(); j++)
		{
			for (int k = 0; k < m1.M(); k++)
			{
				res(i, j) += m1(i, k) * m2(k, j);
			}
		}
	}

	return res;
}



void printMx(const Matrix& m)
{
	std::string prefix = "";
	for (int i = 0; i < m.N(); i++)
	{
		for (int j = 0; j < m.M(); j++)
		{
			std::cout << prefix << m(i, j);
			prefix = " ";
		}
		std::cout << "\n";
		prefix = "";
	}
}

Matrix sigmoidM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		val = 1.0f / (1.0f + expf(val));
	}
	return res;
}

Matrix logM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		val = logf(val);
	}
	return res;
}

//cuts out a part of a Matrix into a new Matrix
Matrix rangeM(const Matrix& m, int i, int j, int w, int h)
{
	Matrix res(h, w);
	for (int k = 0; k < h; k++)
	{
		for (int l = 0; l < w; l++)
		{
			res(k, l) = m(k + i, l + j);
		}
	}
	return res;
}

Matrix appendBelowM(const Matrix& m1, const Matrix& m2)
{
	if (m1.M() != m2.M())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}
	Matrix res(m1.N() + m2.N(), m1.M());

	res.values.clear();
	res.values.insert(res.values.end(), m1.values.begin(), m1.values.end());
	res.values.insert(res.values.end(), m2.values.begin(), m2.values.end());

	return res;
}

Matrix onesM(int i, int j)
{
	return Matrix(i, j, 1.0f);
}

