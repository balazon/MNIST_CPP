#pragma once

#include "Matrix.h"

struct ImageClassificationData
{
	Matrix X;
	Matrix y;

	int M, K;
};