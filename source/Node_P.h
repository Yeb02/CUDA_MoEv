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
	// The following 3 arrays are managed by Network, these are only pointers
	// inside pre-allocated storage:

	// Used as the multiplied vector in matrix operations. Layout:
	// input -> children.out
	float* inputArray;

	// Used as the result vector in matrix operations. Layout:
	// output -> children.in 
	float* destinationArray;

#ifdef STDP
	// Same layout as destinationArray, i.e.
	// output ->  children.in 
	float* destinationArray_preSynAvg;
#endif
	*/


	Node_P(Node_G* type, Node_G** nodes, int i, int iC, int* nC, int tNC);

	// Should never be called.
	Node_P() :
		childrenInputV(nullptr, 0), concInputV(nullptr, 0), outputV(nullptr, 0)
	{
		__debugbreak();
//#ifdef STDP
//		destinationArray_preSynAvg = nullptr;
//#endif
//		destinationArray = nullptr;
//		inputArray = nullptr;

		//new (&v) Map<RowVectorXi>(data + 4, 5);
		

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
