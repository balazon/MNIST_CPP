#pragma once


#include <chrono>


class Timer
{
	std::chrono::high_resolution_clock::time_point startTime;
	
	static Timer inst;
public:

	void start();

	unsigned long long endMillisElapsed();

	static Timer& Instance();
};