#pragma once

#include "config.h"
#include "MLP_G.h"


struct MLP_P {

	MLP_G* type;

	std::vector<std::unique_ptr<float[]>> WGrads;

	std::vector<std::unique_ptr<float[]>> BGrads;

	float* output;

	// Layer by layer activations of the network. 
	std::unique_ptr<float[]> activations;

	// gradients
	std::unique_ptr<float[]> delta;


	MLP_P(MLP_G* type);


	// Should never be called
	MLP_P() {
		__debugbreak();
	}

	// Should never be called
	MLP_P(MLP_P&& n) noexcept {
		__debugbreak();
	}

	// Should never be called
	MLP_P(MLP_P& n) {
		__debugbreak();
	}

	// Gradient accumulation is optional. Input and target are unchanged.
	float forward(float* input, float* target, bool accumulateGrad);

	// accumulates the gradient in the dedicated matrices. The gradient of the cost 
	// with respect to the network's output must already be stored at the beginning of the
	// delta array. Returns the gradient of the cost with respect to the network's input.
	float* accumulateGrad();

	// adds the gradients to the type's parameters
	void addGrad(float lr, float regW, float regB);
};