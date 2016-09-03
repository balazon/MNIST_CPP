#pragma once

#include "Matrix.h"

#include <iostream>
#include <algorithm>
#include <functional>
#include <string>
#include <numeric>
#define EPS 1e-6

Matrix::Matrix() : Matrix(1, 1) {}
Matrix::Matrix(int n, int m, float defVal) : n{ n }, m{ m }, values((size_t)(n * m), defVal)
{

}



Matrix::Matrix(int n, int m, std::vector<float>::const_iterator first, std::vector<float>::const_iterator last,
	const std::vector<float>::allocator_type& alloc) : n{ n }, m{ m }, values{first, last, alloc}
{
	values = std::vector<float>(first, last, alloc);
	values.resize((size_t)n * m, 0.0f);
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

void swap(Matrix& m1, Matrix& m2)
{
	using std::swap;

	std::swap(m1.values, m2.values);
	std::swap(m1.n, m2.n);
	std::swap(m1.m, m2.m);
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

Matrix mulFirstWithSecondTransposedM(const Matrix& m1, const Matrix& m2)
{
	if (m1.M() != m2.M())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}

	Matrix res{ m1.N(), m2.N() };

#pragma omp parallel for
	for (int i = 0; i < res.N(); i++)
	{
		for (int j = 0; j < res.M(); j++)
		{
			float sum = 0.0f;
			for (int k = 0; k < m1.M(); k++)
			{
				sum += m1(i, k) * m2(j, k);
			}
			res(i, j) = sum;
		}
	}

	return res;
}

Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	// transpose second first for improved cache efficiency
	Matrix m2T = m2.transpose();

	return mulFirstWithSecondTransposedM(m1, m2T);
}

Matrix operator==(const Matrix& m1, const Matrix& m2)
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
			res(i, j) = fabs(m1(i, j) - m2(i, j)) < EPS ? 1.0f : 0.0f;
		}
	}

	return res;
}

Matrix mulElementWiseM(const Matrix& m1, const Matrix& m2)
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
			res(i, j) = m1(i, j) * m2(i, j);
		}
	}

	return res;
}


Matrix operator+(const Matrix& m, float val)
{
	Matrix res = m;

	for (int i = 0; i < m.N() * m.M(); i++)
	{
		res(i) = m(i) + val;
	}
	return res;
}

Matrix operator-(const Matrix& m, float val)
{
	Matrix res = m;

	for (int i = 0; i < m.N() * m.M(); i++)
	{
		res(i) = m(i) - val;
	}
	return res;
}

Matrix operator*(const Matrix& m, float val)
{
	Matrix res = m;

	for (int i = 0; i < m.N() * m.M(); i++)
	{
		res(i) = m(i) * val;
	}
	return res;
}

Matrix operator/(const Matrix& m, float val)
{
	if (fabs(val) < EPS)
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}
	Matrix res = m;

	for (int i = 0; i < m.N() * m.M(); i++)
	{
		res(i) = m(i) / val;
	}
	return res;
}


Matrix sigmoidM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		val = 1.0f / (1.0f + expf(-val));
	}
	return res;
}

Matrix sigmoidGradientM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		float sigm = 1.0f / (1.0f + expf(-val));
		val = sigm * (1.0f - sigm);
	}
	return res;
}

Matrix tanhM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		val = tanhf(val);
	}
	return res;
}

Matrix tanhGradientM(const Matrix& m)
{
	Matrix res = m;
	for (float& val : res.values)
	{
		float temp = tanhf(val);
		val = 1.0f - temp * temp;
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
	res.values.insert(res.values.end(), m1.values.cbegin(), m1.values.cend());
	res.values.insert(res.values.end(), m2.values.cbegin(), m2.values.cend());

	return res;
}

void copyMatInM(const Matrix& source, Matrix& dest, int i, int j)
{
	for (int k = 0; k < source.N(); k++)
	{
		for (int l = 0; l < source.M(); l++)
		{
			dest(k + i, l + j) = source(k, l);
		}
	}
}

Matrix appendNextToM(const Matrix& m1, const Matrix& m2)
{
	if (m1.N() != m2.N())
	{
		// error..
		std::cout << "error\n";
		return Matrix();
	}
	Matrix res{ m1.N(), m1.M() + m2.M() };
	copyMatInM(m1, res, 0, 0);
	copyMatInM(m2, res, 0, m1.M());
	return res;
}

Matrix reshapeM(const Matrix& thetas, int startIndex, int n, int m)
{
	Matrix res{ n, m };
	res.values.clear();
	res.values.insert(res.values.end(), thetas.values.cbegin() + startIndex, thetas.values.cbegin() + startIndex + n * m);
	return res;
}

Matrix unrollAllM(const std::vector<Matrix>& matrices)
{
	int count = 0;
	for (int i = 0; i < matrices.size(); i++)
	{
		count += (int)matrices[i].values.size();
	}
	Matrix res{ 1, count };
	res.values.clear();

	for (int i = 0; i < matrices.size(); i++)
	{
		const Matrix& m = matrices[i];
		res.values.insert(res.values.end(), m.values.cbegin(), m.values.cend());
	}

	return res;
}

Matrix zeros(int i, int j)
{
	return Matrix(i, j, 0.0f);
}

Matrix onesM(int i, int j)
{
	return Matrix(i, j, 1.0f);
}

Matrix maxIndexByRowsM(const Matrix& m)
{
	Matrix res{ m.N(), 1 };

	for (int i = 0; i < m.N(); i++)
	{
		float maxVal = FLT_MIN;
		int maxIndex = 0;
		for (int j = 0; j < m.M(); j++)
		{
			float t = m(i, j);
			if (t > maxVal)
			{
				maxVal = t;
				maxIndex = j;
			}
		}
		res(i) = (float)maxIndex;
	}

	return res;
}

Matrix sumByRowsM(const Matrix& m)
{
	Matrix res{ m.N(), 1 };
	int w = m.M();
	for (int i = 0; i < m.N(); i++)
	{
		res(i) = std::accumulate(m.values.cbegin() + i * w, m.values.cbegin() + (i + 1) * w, 0.0f, std::plus<float>());
	}
	return res;
}

float sumAllM(const Matrix& m)
{
	return std::accumulate(m.values.cbegin(), m.values.cend(), 0.0f, std::plus<float>());
}

float meanAllM(const Matrix& m)
{
	return sumAllM(m) / (float)(m.N() * m.M());
}

float sumSquaredAllM(const Matrix& m)
{
	return std::inner_product(m.values.cbegin(), m.values.cend(), m.values.cbegin(), 0.0f);
}

float standardDevM(const Matrix& m, float mean)
{
	if (m.values.size() == 1)
	{
		return 0.0f;
	}
	const std::vector<float>& v = m.values;
	std::vector<float> diff(v.size());
	std::transform(v.cbegin(), v.cend(), diff.begin(), [mean](float x) { return x - mean; });

	float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);

	float stdev = sqrtf(sq_sum / (v.size() - 1.0f));

	return stdev;
}

float standardDevM(const Matrix& m)
{
	float mean = meanAllM(m);
	return standardDevM(m, mean);
}

Matrix normalizeM(const Matrix& m, float& mean, float& stdev)
{
	Matrix res = m;

	mean = meanAllM(m);
	stdev = standardDevM(m, mean);

	res = res - mean;
	res = res / stdev;

	return res;
}