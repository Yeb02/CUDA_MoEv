#pragma once

#include "MLP_P.h"

MLP_P::MLP_P(MLP_G* _type) :
	type(_type)
{

	int activationS = 0;
	for (int i = 0; i < type->nLayers+1; i++) {
		activationS += type->sizes[i];
	}
	activations = std::make_unique<float[]>(activationS);
	delta = std::make_unique<float[]>(activationS);

	output = activations.get() + activationS - type->sizes[type->nLayers];

	WGrads.reserve(type->nLayers);
	BGrads.reserve(type->nLayers);

	for (int i = 0; i < type->nLayers; i++)
	{
		int sW = type->sizes[i] * type->sizes[i + 1];
		WGrads.emplace_back(new float[sW]);
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);

		int sB = type->sizes[i + 1];
		BGrads.emplace_back(new float[sB]);
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}
}


float MLP_P::forward(float* input, float* target, bool accGrad)
{

	float* prevActs = &activations[0];

	std::copy(input, input + type->sizes[0], prevActs);


	float* currActs = &activations[type->sizes[0]];
	for (int i = 0; i < type->nLayers; i++) {
		for (int j = 0; j < type->sizes[i + 1]; j++) {
			currActs[j] = type->Bs[i][j];
		}

		int matID = 0;
		for (int j = 0; j < type->sizes[i + 1]; j++) {
			for (int k = 0; k < type->sizes[i]; k++) {
				currActs[j] += type->Ws[i][matID] * prevActs[k];
				matID++;
			}
		}

		for (int j = 0; j < type->sizes[i + 1]; j++) {
			currActs[j] = tanhf(currActs[j]); // Could be in the previous loop, but GPUisation.
		}

		prevActs = currActs;
		currActs = currActs + type->sizes[i + 1];
	}

	// at this point, prevActs = output.

	float loss = 0.0f; 

	if (target != nullptr) {
		for (int i = 0; i < type->sizes[type->nLayers]; i++) // euclidean distance loss
		{
			float l = powf(output[i] - target[i], 2.0f);
			loss += l;
		}
	}
	

	if (accGrad) {

		for (int i = 0; i < type->sizes[type->nLayers]; i++) 
		{
			delta[i] = (output[i] - target[i]) * (1.0f - output[i] * output[i]);
		}

		accumulateGrad();
	}

	return loss;
}


float* MLP_P::accumulateGrad()
{
	// backward. We will use tanh'(x) = 1 - tanh(x)², most stable and re-uses forward's calculations. 
	// This way presynaptic activations need not be stored.

	float* prevDelta;
	float* currDelta = &delta[0];
	float* currActs = output;
	float* prevActs = currActs - type->sizes[type->nLayers - 1];


	for (int i = type->nLayers - 1; i >= 0; i--) {

		// w
		int matID = 0;
		for (int j = 0; j < type->sizes[i+1]; j++) {
			for (int k = 0; k < type->sizes[i]; k++) {
				WGrads[i][matID] += currDelta[j] * prevActs[k];
				matID++;
			}
		}

		// b
		for (int j = 0; j < type->sizes[i+1]; j++)
		{
			BGrads[i][j] += currDelta[j];
		}

		prevDelta = currDelta;
		currDelta = currDelta + type->sizes[i + 1];

		currActs = prevActs;
		prevActs = currActs - type->sizes[i - 1];

		for (int j = 0; j < type->sizes[i]; j++) {
			currDelta[j] = 0.0f;
		}

		// Update deltas. This way of seeing the matmul avoids non-trivial indices induced by the transposition.
		matID = 0;
		for (int j = 0; j < type->sizes[i + 1]; j++) {
			for (int k = 0; k < type->sizes[i]; k++) {
				currDelta[k] += type->Ws[i][matID + k] * prevDelta[j];
			}
			matID += type->sizes[i];
		}
		for (int j = 0; j < type->sizes[i]; j++) {
			currDelta[j] *= (1.0f - currActs[j] * currActs[j]);
		}

	}

	return currDelta;
}


void MLP_P::addGrad(float lr, float regW, float regB)
{
	for (int i = 0; i < type->nLayers; i++)
	{
		int sW = type->sizes[i] * type->sizes[i + 1];
		for (int j = 0; j < sW; j++) {
			type->Ws[i][j] = type->Ws[i][j] * (1.0f - regW) - WGrads[i][j] * lr;
		}
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);

		int sB = type->sizes[i + 1];
		for (int j = 0; j < sB; j++) {
			type->Bs[i][j] = type->Bs[i][j] * (1.0f - regB) - BGrads[i][j] * lr;
		}
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}
}