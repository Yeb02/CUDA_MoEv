#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "MoEvCore.h"
#include "HebbianNode_G.h"
#include "InternalConnexion_P.h"

struct HebbianNode_P {

	HebbianNode_G* type;

	std::vector<HebbianNode_P> children;


	// Concatenated children inputs. Once computed, each chunk still
	// must be copied to the corresponding child's input array.
	MVector childrenInputV;

	// Concatenation of this node's input and children's output. Once children
	// have computed their output, it must be copied back into this vector.
	MVector concInputV;

	// This node's output.
	MVector outputV;


	InternalConnexion_P toChildren, toOutput;
	

	/*
	// The following 3 arrays are managed by HebbianNetwork, these are only pointers
	// inside pre-allocated storage:

	// Used as the multiplied vector in matrix operations. Layout:
	// input -> children.out
	float* inCoutActivations;

	// Used as the result vector in matrix operations. Layout:
	// output -> children.in 
	float* outCinActivations;

#ifdef STDP
	// Same layout as outCinActivations, i.e.
	// output ->  children.in 
	float* outCinActivations_preSynAvg;
#endif
	*/


	HebbianNode_P(HebbianNode_G* type, HebbianNode_G** nodes, int i, int iC, int* nC, int tNC);

	// Should never be called.
	HebbianNode_P() :
		childrenInputV(nullptr, 0), concInputV(nullptr, 0), outputV(nullptr, 0)
	{
		__debugbreak();
//#ifdef STDP
//		outCinActivations_preSynAvg = nullptr;
//#endif
//		outCinActivations = nullptr;
//		inCoutActivations = nullptr;

		

		type = nullptr;
	}

	~HebbianNode_P() {};

	void preTrialReset();


	void forward();


	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** iA, float** dA, float** dA_preAvg);

};
