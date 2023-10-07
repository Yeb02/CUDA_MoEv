#pragma once

#include "Network.h"
#include "ModulePopulation.h"
#include <iostream>


int Network::destinationArraySize = 0;
int Network::inputArraySize = 0;
int* Network::inS = nullptr;
int* Network::outS = nullptr;
int* Network::nC = nullptr;
int Network::nLayers = 0;



Network::Network(int nModules)
	: IAgent(nModules)
{
	// Quantites created in createdPhenotype:
	topNodeP.reset(NULL);
	inputArray.reset(NULL);
	destinationArray.reset(NULL);
#ifdef STDP
	destinationArray_preSynAvg.reset(NULL);
#endif
}


float* Network::getOutput() 
{
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


void Network::createPhenotype(std::vector<ModulePopulation<class Node_G>*>& populations) 
{

	if (topNodeP.get() != NULL)
	{
		std::cerr << "Called createPhenotype on a Network that already had a phenotype !" << std::endl;
		return;
	}

	int mID = 0;
	int nCpL = 1;
	for (int l = 0; l < nLayers; l++) 
	{
		for (int c = 0; c < nCpL; c++) 
		{
			modules[mID] = static_cast<IModule*>(populations[l]->sample());
			mID++;
		}
		nCpL *= nC[l];
	}


	// I do not understand why I have to use reinterpret_cast instead of static_cast here.
	// Is there a bug of some sort ? TODO 
	topNodeP.reset(new Node_P(static_cast<Node_G*>(modules[0]), reinterpret_cast<Node_G**>(&(modules[0])), 0, 1, nC, 1));

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
	
};


void Network::preTrialReset() {
	nExperiencedTrials++;

	std::fill(inputArray.get(), inputArray.get() + inputArraySize, 0.0f);

	topNodeP->preTrialReset();
};


void Network::step(float* input) {
	std::copy(input, input + inS[0], topNodeP->inputArray);
	
	topNodeP->forward();

	nInferencesOverLifetime++;
}


void Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	// TODO. Write the phenotypic parameters and references to the genotypic ones.

}


Network::Network(std::ifstream& is) :
	IAgent(is)
{
	/*int version;
	READ_4B(version, is);*/

	// TODO.
}