#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>

#include "VirtualModule.h"
#include "MoEvCore.h"
#include "InternalConnexion_G.h"


class HebbianNode_GFixedParameters 
{
public:
	int nChildren;
	int inputSize, outputSize;

	int nCols;
	int toChildrenNLines;

	HebbianNode_GFixedParameters(int* inS, int* outS, int* nC) :
		nChildren(nC[0]), inputSize(inS[0]), outputSize(outS[0])
	{

		toChildrenNLines = nC[0] > 0 ? outS[1] * nC[0] : 0;
		nCols = nC[0] > 0 ? inS[1] * nC[0] + inputSize : inputSize;
	}
};


class HebbianNode_G : public IModule
{
public:
	HebbianNode_G(HebbianNode_G* n);

	~HebbianNode_G() {};


	int nChildren;
	int inputSize, outputSize; // >= 1

	// Structs containing the constant, evolved, matrix of parameters linking internal nodes.
	// The name specifies the type of node that takes the result of the matrix operations as inputs.
	// nColumns = this.inputSize + MODULATION_VECTOR_SIZE + sum(children.inputSize)
	InternalConnexion_G toChildren; // nRows = sum(children.inputSize) 
	InternalConnexion_G toOutput; // nRows = outputSize


	int getNParameters() override {return toChildren.getNParameters() + toOutput.getNParameters();}

	HebbianNode_G(std::ifstream& is);

	HebbianNode_G(HebbianNode_GFixedParameters& p);

	void save(std::ofstream& os) override;

	void mutate(float p) override;

	static HebbianNode_G* combine(HebbianNode_G** parents, float* weights, int nParents);
};

