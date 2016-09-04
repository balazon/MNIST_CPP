

#include "Timer.h"


Timer Timer::inst = Timer{};

void Timer::start()
{
	startTime = std::chrono::high_resolution_clock::now();
}

unsigned long long Timer::endMillisElapsed()
{
	auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
	
	return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

Timer& Timer::Instance()
{
	return inst;
}