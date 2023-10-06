#pragma once


//#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include <curand.h>

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "VirtualClasses.h"
#include "Random.h"
#include "Node_P.h"
#include "Node_G.h"
#include "ModulePopulation.h"
//#include "MLP_G.h"
//#include "MLP_P.h"



struct NetworkParameters 
{
	// How deep the networks are
	int nLayers;

	// layer by layer input sizes of the modules.
	int* inSizes;

	// layer by layer output sizes of the modules.
	int* outSizes;

	// layer by layer number of children per module. Is 0 only at nLayers'th position.
	int* nChildrenPerLayer;

	NetworkParameters() {};
};


class Network : public IAgent {
	
public:

	Network(int nModules);

	~Network() {};
	


	static int destinationArraySize;
	static int inputArraySize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

	std::unique_ptr<Node_P> topNodeP;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations. More detail on each array in the Node_P class.
	
	// The contiguous inputs of the nodes
	std::unique_ptr<float[]> inputArray;

	
	std::unique_ptr<float[]> destinationArray;
#ifdef STDP
	std::unique_ptr<float[]> destinationArray_preSynAvg;
#endif


	Network(std::ifstream& is);

	void save(std::ofstream& os) override;

	float* getOutput() override;

	void step(float* input) override;

	void preTrialReset() override;
	
	void createPhenotype(std::vector<ModulePopulation<class Node_G>*>& populations);

	void destroyPhenotype() override;

};