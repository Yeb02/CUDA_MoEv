#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "Random.h"
#include "Node_P.h"
#include "Node_G.h"



class Network {
	
public:
	Network(int* inS, int* outS, int* nC, int depth);

	Network(Network* n);
	~Network() {};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(const std::vector<float>& obs);
	
	void createPhenotype(int inputArraySize, int destinationArraySize, Node_G** nodes);
	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

private:

	int destinationArraySize, inputArraySize;

	// These quantities are not strictly necessary to store, I do it for convenience.
	int* inS;
	int* outS;
	int* nC;
	int depth;


	std::unique_ptr<Node_P> topNodeP;


	// The following arrays hold per-neuron quantities for the whole network. 
	// Each node of the tree has its own pointers inside each array to its 
	// dedicated, "personal", storage. Done this way to minimize cache misses
	// and memory allocations. More detail on each array in the Node_P class.
	
	// The contiguous inputs of the nodes
	std::unique_ptr<float[]> inputArray;

	// The contiguous inputs of the nodes, average (moving exponential) over the
	// last inference steps with an evolved decay per neuron.
	std::unique_ptr<float[]> inputArray_avg;

	
	std::unique_ptr<float[]> destinationArray;
	std::unique_ptr<float[]> destinationArray_avg;
#ifdef STDP
	std::unique_ptr<float[]> destinationArray_preAvg;
#endif


	int inputArraySize;
	int destinationArraySize;

	// How many inferences were performed since last call to preTrialReset by the phenotype.
	int nInferencesOverTrial;

	// How many inferences were performed since phenotype creation.
	int nInferencesOverLifetime;

	// How many trials the phenotype went through.
	int nExperiencedTrials;
};