#pragma once

#include <functional>
#include <vector>

#include "Matrix.h"

//costFun should be a function returning the cost and the gradient of the function:
// params are theta, and the gradient (as a reference that is set inside the cost function)
// return value is the cost (float)
void gradientDescent(std::function<float (const std::vector<float>&, std::vector<float>&)> costFun, std::vector<float>& theta, float alpha, int numIters);


//Matrix overload to avoid unwrap + wrap
void gradientDescent(std::function<float(const Matrix&, Matrix&)> costFun, Matrix& theta, float alpha, int numIters);