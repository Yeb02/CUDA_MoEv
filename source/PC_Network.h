#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "MoEvCore.h"
#include "VirtualAgent.h"
#include "PC_Node_P.h"
#include "PC_Node_G.h"
#include "ModulePopulation.h"




class PC_NetworkParameters : public INetworkFixedParameters
{
public:
	PC_NetworkParameters() {};
};


class PC_Network : public IAgent {
	
public:

	PC_Network(int nModules);

	// WARNING this constructor does not create a valid agent for evolution, but merely 
	// a "teacher" agent. It should NOT be used to create anything else than a teacher.
	PC_Network(const PC_Network& pcn);

	~PC_Network() {};
	


	static int activationArraySize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

	std::unique_ptr<PC_Node_P> rootNode;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations.
	std::unique_ptr<float[]> activations;
	std::unique_ptr<float[]> accumulators;

	void setInitialActivations(float* initialActivations) { std::copy(initialActivations, initialActivations + activationArraySize, activations.get()); };


	PC_Network(std::ifstream& is);

	void save(std::ofstream& os) override;

	float* getOutput() override;

	void step(float* input, bool supervised, float* target = nullptr) override;

	// This has become obsolete, consider removing it. Problem is that it is useful for hebbian networks.
	void preTrialReset() override;
	
	void createPhenotype(std::vector<ModulePopulation*>& populations);

	void destroyPhenotype() override;

};