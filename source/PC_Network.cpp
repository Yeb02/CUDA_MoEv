#pragma once

#include "MoEvCore.h"
#ifdef PREDICTIVE_CODING

#include "PC_Network.h"
#include "ModulePopulation.h"
#include <iostream>


int PC_Network::activationArraySize = 0;
int* PC_Network::inS = nullptr;
int* PC_Network::outS = nullptr;
int* PC_Network::nC = nullptr;
int PC_Network::nLayers = 0;



PC_Network::PC_Network(int nModules)
	: IAgent(nModules)
{
	// Quantites created in createdPhenotype:
	rootNode.reset(NULL);
	activations.reset(NULL);
	accumulators.reset(NULL);
}


PC_Network::PC_Network(const PC_Network& pcn)
	: IAgent(0) 
{

	rootNode.reset(new PC_Node_P(*(pcn.rootNode.get())));

	activations = std::make_unique<float[]>(activationArraySize);
	accumulators = std::make_unique<float[]>(activationArraySize);

	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations.get() + outS[0];
	float* ptr_accumulators = accumulators.get() + outS[0];
	float* outputActivations = activations.get();
	float* outputAccumulators = accumulators.get();

	rootNode->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);
}


float* PC_Network::getOutput()
{
	return rootNode->outputActivations.data();
}


void PC_Network::destroyPhenotype() {
	rootNode.reset(NULL);

	activations.reset(NULL);
	accumulators.reset(NULL);

}


void PC_Network::createPhenotype(std::vector<ModulePopulation*>& populations)
{

	if (rootNode.get() != NULL)
	{
		std::cerr << "Called createPhenotype on a PC_Network that already had a phenotype !" << std::endl;
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
	rootNode.reset(new PC_Node_P(modules[0], &(modules[0]), 0, 1, nC, 1));


	activations = std::make_unique<float[]>(activationArraySize);
	accumulators = std::make_unique<float[]>(activationArraySize);
		
	// The following values will be modified by each node of the phenotype as the pointers are set.
	float* ptr_activations = activations.get() + outS[0];
	float* ptr_accumulators = accumulators.get() + outS[0];
	float* outputActivations = activations.get();
	float* outputAccumulators = accumulators.get();

	rootNode->setArrayPointers(
		&ptr_activations,
		&ptr_accumulators,
		outputActivations,
		outputAccumulators
	);
	
};


void PC_Network::preTrialReset() {
	nExperiencedTrials++;
};


void PC_Network::step(float* input, bool supervised, float* target) 
{
	std::copy(input, input + inS[0], rootNode->inputActivations.data());
	
	if (supervised) {
		std::copy(target, target + outS[0], rootNode->outputActivations.data());
	}

	for (int i = 0; i < 10; i++) 
	{
		std::fill(accumulators.get(), accumulators.get() + activationArraySize, 0.0f);
		rootNode->xUpdate_simultaneous();
		if (!supervised) {
			rootNode->outputActivations += .1f * rootNode->outputAccumulators;
		}
	}

	if (supervised) {
		//std::fill(accumulators.get(), accumulators.get() + activationArraySize, 0.0f);
		rootNode->thetaUpdate_simultaneous();
	}

	nInferencesOverLifetime++;
}


void PC_Network::save(std::ofstream& os)
{
	int version = 0;
	WRITE_4B(version, os); // version

	// TODO. Write the phenotypic parameters and references to the genotypic ones.

}


PC_Network::PC_Network(std::ifstream& is) :
	IAgent(is)
{
	/*int version;
	READ_4B(version, is);*/

	// TODO.
}

#endif