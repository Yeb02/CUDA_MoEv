#pragma once

#include "MoEvCore.h"

#include <vector>
#include <fstream>



struct Expliciter 
{
	std::unique_ptr<float[]> as;
	std::unique_ptr<float[]> bs;
	int n;

	Expliciter(int _n) : n(_n) 
	{
		as = std::make_unique<float[]>(n);
		bs = std::make_unique<float[]>(n);


		// TODO
	}

	// requires the array x has n free slots AFTER x. 
	void enlarge(float* x) {
		for (int i = 0; i < n; i++) {
			*(x + 1 + i) = sinf(as[i] * (*x) + bs[i]);
		}
	}
};

// The base virtual class which any trial should inherit from. 
// The score attribute must be a positive measure of the success of the run.
class Trial {

public:
	// Given the actions of the network, proceeds one step forward in the trial.
	virtual void step(const float* actions) = 0;

	// To be called at the end of the trial, AFTER fetching the score !
	// When sameSeed is true, the random values are kept between runs.
	virtual void reset(bool sameSeed = false) = 0;

	// copies the constant and per-run parameters of t. Must cast to derived class:
	// DerivedTrial* t = dynamic_cast<DerivedTrial*>(t0);
	virtual void copy(Trial* t0) = 0;

	// returns a pointer to a new instance OF THE DERIVED CLASS, cast to a pointer of the base class.
	virtual Trial* clone() = 0;

	// Handle for updates coming from the outer loop (main.cpp 's loop)
	virtual void outerLoopUpdate(void* data) = 0;

	std::vector<float> observations;

	// the required network dimensions
	int netInSize, netOutSize;

	float score;

	virtual ~Trial() = default; // otherwise derived destructors will not be called.

	bool isTrialOver;

protected:

	// the current elapsed steps in the trial. To be set to 0 in reset.
	int currentNStep;
};
