#include "GradientDescent.h"
#include "Matrix.h"

void gradientDescent(std::function<float (const std::vector<float>&, std::vector<float>&)> costFun, std::vector<float>& theta, float alpha, int numIters)
{
	Matrix grad{ 1, (int)theta.size() };
	Matrix thetaMx{ 1, (int)theta.size(), theta.cbegin(), theta.cend() };

	for (int i = 0; i < numIters; i++)
	{
		float cost = costFun(thetaMx.values, grad.values);
		
		thetaMx = thetaMx - grad * alpha;
	}

	theta = thetaMx.values;
}

void gradientDescent(std::function<float(const Matrix&, Matrix&)> costFun, Matrix& theta, float alpha, int numIters)
{
	Matrix grad{ 1, theta.N() * theta.M() };
	printf("Gradient descent:\n");

	for (int i = 0; i < numIters; i++)
	{
		float cost = costFun(theta, grad);
		printf("  cost: %f, iter: %d\n", cost, (i + 1));
		theta = theta - grad * alpha;
	}
}