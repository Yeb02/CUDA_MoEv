#pragma once

#include "MoEvCore.h"

#include <vector>
#include <fstream>

#ifdef PREDICTIVE_CODING
#include "PC_Node_G.h"
#else
#include "HebbianNode_G.h"
#endif



// To give a common base to parameters used to create agents (that are modeled as networks)
class INetworkFixedParameters
{
public:
	// How deep the networks are
	int nLayers;

	// layer by layer input sizes of the modules.
	int* inSizes;

	// layer by layer output sizes of the modules.
	int* outSizes;

	// layer by layer number of children per module. Is 0 only at nLayers'th position.
	int* nChildrenPerLayer;
};




class IAgent 
{
	
public:

	int nExperiencedTrials;

	float lifetimeFitness;

	int nModules;

	int nInferencesOverLifetime;

	std::unique_ptr<MODULE* []> modules;

	IAgent(std::ifstream& is) 
	{
		READ_4B(nModules, is);
		modules = std::make_unique<MODULE * []>(nModules);
	}
	
	IAgent(int _nModules)
	{
		nExperiencedTrials = 0;
		nInferencesOverLifetime = 0;
		lifetimeFitness = 0.0f;
		nModules = _nModules;
		modules = std::make_unique<MODULE * []>(nModules);
	}

	virtual ~IAgent() 
	{
		for (int i = 0; i < nModules; i++) modules[i]->nUsesInAgents--;
	};

	void accumulateFitnessInModules(float f)
	{
		for (int j = 0; j < nModules; j++) {
			modules[j]->tempFitnessAccumulator += f;
			modules[j]->nTempFitnessAccumulations++;
		}
	}


	// Same as for the Module class. LAME
	// virtual Agent(std::ifstream& is) = 0; TODO find a way to do this...
	//virtual void createPhenotype(std::vector<ModulePopulation<class Module>*>& populations) = 0; An agent need not have a phenotype, in which case the overrides are empty
	// //class ModulePopulation; need be forward declarezd.

	virtual void save(std::ofstream& os) = 0;

	virtual float* getOutput() = 0;

	virtual void step(float* input, bool supervised, float* target = nullptr) = 0;

	// Probably empty, but in rare cases it might be used.
	virtual void preTrialReset() = 0;

	virtual void destroyPhenotype() = 0;

};