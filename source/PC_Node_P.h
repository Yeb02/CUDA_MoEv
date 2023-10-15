#pragma once

#include <vector>
#include <memory>
#include <cmath>

#include "MoEvCore.h"
#include "PC_Node_G.h"
#include "InternalConnexion_P.h"

struct PC_Node_P {

	PC_Node_G* type;

	std::vector<PC_Node_P> children;


	// Concatenation of this node's input and children's output. 
	MVector inCoutActivations;
	MVector inCoutAccumulators;

	// This node's input
	MVector inputActivations;
	MVector inputAccumulators;

	// This node's output.
	MVector outputActivations;
	MVector outputAccumulators;



	InternalConnexion_P toOutput;
	std::vector<InternalConnexion_P> toChildren;
	


	PC_Node_P(PC_Node_G* type, PC_Node_G** nodes, int i, int iC, int* nC, int tNC);

	// Should never be called.
	PC_Node_P() :
		inCoutActivations(nullptr, 0), outputActivations(nullptr, 0), inputActivations(nullptr, 0),
		inCoutAccumulators(nullptr, 0), outputAccumulators(nullptr, 0), inputAccumulators(nullptr, 0)
	{
		__debugbreak();


		type = nullptr;
	}

	~PC_Node_P() {};

	void preTrialReset() {};

	// Recursive function. When the network calls it on the root node, all activations of the tree are updated
	// simultaneously.
	void xUpdate_simultaneous();

	// Recursive function. When the network calls it on the root node, all weights, biases, and modulation weights
	// of the tree are updated simultaneously.
	void thetaUpdate_simultaneous();


	void setArrayPointers(float** ptr_activations, float** ptr_accumulators, float* outActivations, float* outAccumulators);

};
