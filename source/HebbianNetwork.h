#pragma once


#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "VirtualAgent.h"
#include "MoEvCore.h"
#include "HebbianNode_P.h"
#include "HebbianNode_G.h"
#include "ModulePopulation.h"



class HebbianNetworkParameters : public INetworkFixedParameters
{
public:
	HebbianNetworkParameters() {};
};

class HebbianNetwork : public IAgent {
	
public:

	HebbianNetwork(int nModules);

	~HebbianNetwork() {};
	


	static int outCinSize;
	static int inCoutSize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

	std::unique_ptr<HebbianNode_P> rootNode;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations. More detail on each array in the HebbianNode_P class.
	
	// The contiguous inputs of the nodes
	std::unique_ptr<float[]> inCoutActivations;

	
	std::unique_ptr<float[]> outCinActivations;
#ifdef STDP
	std::unique_ptr<float[]> outCinActivations_preSynAvg;
#endif


	HebbianNetwork(std::ifstream& is);

	void save(std::ofstream& os) override;

	float* getOutput() override;

	void step(float* input, bool supervised, float* target = nullptr) override;

	void preTrialReset() override;
	
	void createPhenotype(std::vector<ModulePopulation*>& populations);

	void destroyPhenotype() override;

};