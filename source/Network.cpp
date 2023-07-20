#pragma once

#include "Network.h"
#include <iostream>

Network::Network(Network* n)
{
	
	inS = n->inS;
	outS = n->outS;
	nC = n->nC; 
	depth = n->depth;

	inputArraySize = -1;
	destinationArraySize = -1;

	topNodeP.reset(NULL);
}


Network::Network(int* inS, int* outS, int* nC, int depth) :
	inS(inS), outS(outS), nC(nC), depth(depth)
{
	inputArraySize = -1;
	destinationArraySize = -1;

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


void Network::createPhenotype(int _inputArraySize, int _destinationArraySize, Node_G** nodes) {
	if (topNodeP.get() == NULL) {
		inputArraySize = _inputArraySize;
		destinationArraySize = _destinationArraySize;

		topNodeP.reset(new Node_P(nodes[0], nodes, 0, 1, nC, 1));

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
};


void Network::step(const std::vector<float>& obs) {

	// float* kappa = topNodeP  aie aie aie TODO TODO TODO PRIORITAIRE
	for (int i = 0; i < obs.size(); i++) {
		topNodeP->inputArray_avg[i] = topNodeP->inputArray_avg[i]*.9f + topNodeP->inputArray[i] * .1f;
	}

	bool firstCall = nInferencesOverTrial == 0;

	if (firstCall) [[unlikely]]
	{
		std::copy(obs.begin(), obs.end(), topNodeP->inputArray_avg);
	}

	std::copy(obs.begin(), obs.end(), topNodeP->inputArray);

	std::fill(topNodeP->totalM, topNodeP->totalM + MODULATION_VECTOR_SIZE, 0.0f);

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