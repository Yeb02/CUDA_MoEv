#pragma once

#include "Network.h"
#include <iostream>


int Network::destinationArraySize = 0;
int Network::inputArraySize = 0;
int* Network::inS = nullptr;
int* Network::outS = nullptr;
int* Network::nC = nullptr;
int Network::nLayers = 0;

float**** Network::mats_CUDA = nullptr;
float**** Network::vecs_CUDA = nullptr;



Network::Network(int nTrialsPerGroup)
{
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
	destinationArray.reset(NULL);

#ifdef STDP
	destinationArray_preSynAvg.reset(NULL);
#endif

}


void Network::createPhenotype(Node_G** _nodes) {
	if (topNodeP.get() == NULL) {
		nodes.reset(_nodes);

		topNodeP.reset(new Node_P(nodes[0], nodes.get(), 0, 1, nC, 1));

		destinationArray = std::make_unique<float[]>(destinationArraySize);
		
#ifdef STDP
		destinationArray_preSynAvg = std::make_unique<float[]>(destinationArraySize);
#endif

		inputArray = std::make_unique<float[]>(inputArraySize);

		
		// The following values will be modified by each node of the phenotype as the pointers are set.
		float* ptr_iA = inputArray.get();
		float* ptr_dA = destinationArray.get();

#ifdef STDP
		float* ptr_dA_preSynAvg = destinationArray.get();
#else
		float* ptr_dA_preSynAvg = nullptr;
#endif

		topNodeP->setArrayPointers(
			&ptr_iA,
			&ptr_dA,
			&ptr_dA_preSynAvg
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


void Network::step(float* input) {

	std::copy(input, input+inS[0], topNodeP->inputArray);

	topNodeP->forward();

	nInferencesOverLifetime++;
	nInferencesOverTrial++;
}



void Network::uploadToGPU(int netId) {

}

void Network::randomizeH(int netId) {

}

void Network::retrieveLearnedParametersFromGPU(int netId)
{

}

void Network::grouped_step(Network** nets, int nNets) 
{

}

void Network::grouped_perLayer_Forward(int layer)
{

}

void Network::grouped_perDestination_propagateAndLocalUpdate(int destination)
{

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