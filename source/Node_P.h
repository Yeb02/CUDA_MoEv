#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "Random.h"
#include "Node_G.h"
#include "InternalConnexion_P.h"

struct Node_P {
	Node_G* type;

	float totalM[MODULATION_VECTOR_SIZE]; // parent's + local M.    

	 
	std::vector<Node_P> children;

	InternalConnexion_P toChildren, toModulation, toOutput;
	

	// These arrays are not managed by Complex node, but by Network:
	 


	// Used as the multiplied vector in matrix operations. Layout:
	// input -> modulation.out -> children.out
	float* inputArray;
	float* inputArray_avg;

	// Used as the result vector in matrix operations. Layout:
	// output -> modulation.in -> children.in 
	float* destinationArray;
	float* destinationArray_avg;

#ifdef STDP
	// Same layout as destinationArray, i.e.
	// output -> modulation.in -> children.in 
	float* destinationArray_preAvg;
#endif



	Node_P(Node_G* type, Node_G** nodes, int i, int iC, int* nC, int tNC);

	// Should never be called.
	Node_P() 
	{
		__debugbreak();
#ifdef STDP
		destinationArray_preAvg = nullptr;
#endif
		destinationArray_avg = nullptr;
		destinationArray = nullptr;
		inputArray_avg = nullptr;
		inputArray = nullptr;
		std::fill(totalM, totalM + MODULATION_VECTOR_SIZE, 0.0f);
		type = nullptr;
	}

	~Node_P() {};

	void preTrialReset();

	void forward(bool firstCall);


	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** iA, float** iA_avg, float** dA, float** dA_avg, float** dA_preAvg);

};
