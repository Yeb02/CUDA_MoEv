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
	// input -> modulation.out -> children.out -> memoryChildren.out
	float* postSynActs;

	// Used as the result vector in matrix operations. Layout:
	// output -> modulation.in -> children.in -> memoryChildren.in
	float* preSynActs;

#ifdef STDP
	// Same layout as PreSynActs, i.e.
	// output -> modulation.in -> children.in -> memoryChildren.in
	float* accumulatedPreSynActs;
#endif

#ifdef SATURATION_PENALIZING
	// Layout:
	// Modulation -> (children->inputSize) -> (memoryChildren->inputSize (mn owns it))
	float* averageActivation;

	// A parent updates it for its children (in and out), not for itself.
	float* globalSaturationAccumulator;
#endif


	Node_P(Node_G* type);

	// Should never be called.
	Node_P() 
	{
		__debugbreak();
#ifdef STDP
		accumulatedPreSynActs = nullptr;
#endif
#ifdef SATURATION_PENALIZING
		averageActivation = nullptr;
		globalSaturationAccumulator = nullptr;
#endif
		preSynActs = nullptr;
		postSynActs = nullptr;
		std::fill(totalM, totalM + MODULATION_VECTOR_SIZE, 0.0f);
		type = nullptr;
	}

	~Node_P() {};

	void preTrialReset();

	void forward(bool firstCall);


	// The last 2 parameters are optional :
	// - aa only used when SATURATION_PENALIZING is defined
	// - acc_pre_syn_acts only used when STDP is defined
	void setArrayPointers(float** pre_syn_acts, float** post_syn_acts, float** aa, float** acc_pre_syn_acts);

#ifdef SATURATION_PENALIZING
	void setglobalSaturationAccumulator(float* globalSaturationAccumulator);
#endif
};
