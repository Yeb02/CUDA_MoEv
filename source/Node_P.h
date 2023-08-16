#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Node_G.h"
#include "InternalConnexion_P.h"

struct Node_P {
	Node_G* type;

	 
	std::vector<Node_P> children;

	InternalConnexion_P toChildren, toOutput;
	

	// These arrays are not managed by Complex node, but by Network:
	 


	// Used as the multiplied vector in matrix operations. Layout:
	// input -> modulation.out -> children.out
	float* inputArray;

	// Used as the result vector in matrix operations. Layout:
	// output -> modulation.in -> children.in 
	float* destinationArray;

#ifdef STDP
	// Same layout as destinationArray, i.e.
	// output -> modulation.in -> children.in 
	float* destinationArray_preSynAvg;
#endif



	Node_P(Node_G* type, Node_G** nodes, int i, int iC, int* nC, int tNC);

	// Should never be called.
	Node_P() 
	{
		__debugbreak();
#ifdef STDP
		destinationArray_preSynAvg = nullptr;
#endif
		destinationArray = nullptr;
		inputArray = nullptr;
		type = nullptr;
	}

	~Node_P() {};

	void preTrialReset();


	void forward();


	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** iA, float** dA, float** dA_preAvg);

};
