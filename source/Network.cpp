#pragma once

#include "Network.h"
#include <iostream>

Network::Network(int* inS, int* outS, int* nC, int depth, int nTrialsPerGroup) :
	inS(inS), outS(outS), nC(nC), depth(depth)
{
	inputArraySize = -1;
	destinationArraySize = -1;

	lifetimeFitness = 0.0f;
	
	groupID = -1;
	perTrialVotes = std::make_unique<float[]>(nTrialsPerGroup);

	topNodeP.reset(NULL);
}


float* Network::getOutput() {
	return topNodeP->destinationArray;
}


void Network::destroyPhenotype() {
	topNodeP.reset(NULL);

	inputArray.reset(NULL);
	inputArray_avg.reset(NULL);
	destinationArray.reset(NULL);
	destinationArray_avg.reset(NULL);

#ifdef STDP
	destinationArray_preAvg.reset(NULL);
#endif

}


void Network::createPhenotype(int _inputArraySize, int _destinationArraySize, Node_G** _nodes) {
	if (topNodeP.get() == NULL) {
		nodes.reset(_nodes);

		inputArraySize = _inputArraySize;
		destinationArraySize = _destinationArraySize;

		topNodeP.reset(new Node_P(nodes[0], nodes.get(), 0, 1, nC, 1));

		destinationArray = std::make_unique<float[]>(destinationArraySize);
		destinationArray_avg = std::make_unique<float[]>(destinationArraySize);
		
#ifdef STDP
		destinationArray_preAvg = std::make_unique<float[]>(destinationArraySize);
#endif

		inputArray = std::make_unique<float[]>(inputArraySize);
		inputArray_avg = std::make_unique<float[]>(inputArraySize);

		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* ptr_iA = inputArray.get();
		float* ptr_iA_avg = inputArray.get();
		float* ptr_dA = destinationArray.get();
		float* ptr_dA_avg = destinationArray.get();

#ifdef STDP
		float* ptr_dA_preAvg = destinationArray.get();
#else
		float* ptr_dA_preAvg = nullptr;
#endif

		topNodeP->setArrayPointers(
			&ptr_iA,
			&ptr_iA_avg,
			&ptr_dA,
			&ptr_dA_avg,
			&ptr_dA_preAvg
		);

		nInferencesOverTrial = 0;
		nInferencesOverLifetime = 0;
		nExperiencedTrials = 0;
	}
};


void Network::preTrialReset() {
	nInferencesOverTrial = 0;
	nExperiencedTrials++;

	std::fill(inputArray.get(), inputArray.get() + inputArraySize, 0.0f);

	topNodeP->preTrialReset();

	std::fill(topNodeP->totalM, topNodeP->totalM + MODULATION_VECTOR_SIZE, 1.0f);
};


void Network::step(float* input) {

	// float* kappa = topNodeP  aie aie aie TODO TODO TODO PRIORITAIRE
	for (int i = 0; i < inS[0]; i++) {
		topNodeP->inputArray_avg[i] = topNodeP->inputArray_avg[i]*.8f + topNodeP->inputArray[i] * .2f;
	}

	bool firstCall = (nInferencesOverTrial == 0);

	if (firstCall) [[unlikely]]
	{
		std::copy(input, input + inS[0], topNodeP->inputArray_avg);
	}

	std::copy(input, input+inS[0], topNodeP->inputArray);

	// On cartpole, the 0.0f version yields the best results. But I think that both
	// should be commented on a real trial, and replaced by the line below.
	//std::fill(topNodeP->totalM, topNodeP->totalM + MODULATION_VECTOR_SIZE, 1.0f);
	//std::fill(topNodeP->totalM, topNodeP->totalM + MODULATION_VECTOR_SIZE, 0.0f); 
	// Replaced by:
	for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) topNodeP->totalM[i] *= .7f;


	topNodeP->forward(firstCall);

	nInferencesOverLifetime++;
	nInferencesOverTrial++;
}


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	// TODO. Write the phenotypic parameters and references to the genotypic ones.

}

Network::Network(std::ifstream& is)
{
	int version;
	READ_4B(version, is);
	// TODO.
}