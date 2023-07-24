#pragma once


#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <vector>
#include <memory>
#include <cmath>
#include <fstream>

#include "Random.h"
#include "Node_P.h"
#include "Node_G.h"



class Network {
	
public:
	Network(int nTrialsPerGroup);

	~Network() {};

	Network(std::ifstream& is);
	
	void save(std::ofstream& os);

	// Since getOutput returns a float*, application must either use it before any other call to step(),
	// destroyPhenotype(), preTrialReset(), ... or Network destruction, either deep copy the
	// pointee immediatly when getOutput() returns. If unsure, deep copy.
	float* getOutput();

	void step(float* input);


	//**************************		CUDA		*************************//

	void uploadToGPU(int netId);
	void randomizeH(int netId);
	void retrieveLearnedParametersFromGPU(int netId);

	static void grouped_step(Network** nets, int nNets);
	static void grouped_perLayer_Forward(int layer);
	static void grouped_perDestination_propagateAndLocalUpdate(int destination);

	static float**** mats_CUDA;
	static float**** vecs_CUDA;

	//***********************************************************************//
	

	void createPhenotype(Node_G** nodes);
	void destroyPhenotype();

	// Sets to 0 the dynamic elements of the phenotype. 
	void preTrialReset();

	// How many trials the phenotype went through.
	int nExperiencedTrials;

	float lifetimeFitness;

	// The ID of the group this network is part of.
	int groupID;

	// Stores the votes on each trial the group experiences. When a new group is formed, these are overwritten.
	// Used by the population for computing fitnesses.
	std::unique_ptr<float[]> perTrialVotes;

	// Breadth first traversal of the tree. Used by population for module score calculations.
	std::unique_ptr<Node_G* []> nodes;


	static int destinationArraySize;
	static int inputArraySize;
	static int* inS;
	static int* outS;
	static int* nC;
	static int nLayers;

private:

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

	// How many inferences were performed since last call to preTrialReset by the phenotype.
	int nInferencesOverTrial;

	// How many inferences were performed since phenotype creation.
	int nInferencesOverLifetime;

};