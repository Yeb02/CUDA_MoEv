#pragma once

#include "MoEvCore.h"
#ifndef PREDICTIVE_CODING



#include "HebbianNetwork.h"
#include "ModulePopulation.h"
#include <iostream>


int HebbianNetwork::outCinSize = 0;
int HebbianNetwork::inCoutSize = 0;
int* HebbianNetwork::inS = nullptr;
int* HebbianNetwork::outS = nullptr;
int* HebbianNetwork::nC = nullptr;
int HebbianNetwork::nLayers = 0;



HebbianNetwork::HebbianNetwork(int nModules)
	: IAgent(nModules)
{
	// Quantites created in createdPhenotype:
	rootNode.reset(NULL);
	inCoutActivations.reset(NULL);
	outCinActivations.reset(NULL);
#ifdef STDP
	outCinActivations_preSynAvg.reset(NULL);
#endif
}


float* HebbianNetwork::getOutput() 
{
	return rootNode->outputV.data();
}


void HebbianNetwork::destroyPhenotype() {
	rootNode.reset(NULL);

	inCoutActivations.reset(NULL);
	outCinActivations.reset(NULL);

#ifdef STDP
	outCinActivations_preSynAvg.reset(NULL);
#endif

}


void HebbianNetwork::createPhenotype(std::vector<ModulePopulation*>& populations) 
{

	if (rootNode.get() != NULL)
	{
		std::cerr << "Called createPhenotype on a HebbianNetwork that already had a phenotype !" << std::endl;
		return;
	}

	int mID = 0;
	int nCpL = 1;
	for (int l = 0; l < nLayers; l++) 
	{
		for (int c = 0; c < nCpL; c++) 
		{
			modules[mID] = populations[l]->sample();
			mID++;
		}
		nCpL *= nC[l];
	}


	// I do not understand why I have to use reinterpret_cast instead of static_cast here.
	// Is there a bug of some sort ? TODO 
	rootNode.reset(new HebbianNode_P(modules[0], &(modules[0]), 0, 1, nC, 1));

	outCinActivations = std::make_unique<float[]>(outCinSize);
		
#ifdef STDP
	outCinActivations_preSynAvg = std::make_unique<float[]>(outCinSize);
#endif

	inCoutActivations = std::make_unique<float[]>(inCoutSize);

		
	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_iA = inCoutActivations.get();
	float* ptr_dA = outCinActivations.get();

#ifdef STDP
	float* ptr_dA_preSynAvg = outCinActivations.get();
#else
	float* ptr_dA_preSynAvg = nullptr;
#endif

	rootNode->setArrayPointers(
		&ptr_iA,
		&ptr_dA,
		&ptr_dA_preSynAvg
	);
	
};


void HebbianNetwork::preTrialReset() {
	nExperiencedTrials++;

	std::fill(inCoutActivations.get(), inCoutActivations.get() + inCoutSize, 0.0f);

	rootNode->preTrialReset();
};


void HebbianNetwork::step(float* input, bool supervised, float* target) {
	std::copy(input, input + inS[0], rootNode->concInputV.data());
	
	rootNode->forward();

	nInferencesOverLifetime++;
}


void HebbianNetwork::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	// TODO. Write the phenotypic parameters and references to the genotypic ones.

}


HebbianNetwork::HebbianNetwork(std::ifstream& is) :
	IAgent(is)
{
	/*int version;
	READ_4B(version, is);*/

	// TODO.
}

#endif