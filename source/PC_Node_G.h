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




class PC_Node_GFixedParameters
{

public:
	int nChildren;
	int inputSize, outputSize;

	int nCols;
	int childrenInputSize;

	PC_Node_GFixedParameters(int* inS, int* outS, int* nC) :
		nChildren(nC[0]), inputSize(inS[0]), outputSize(outS[0])
	{

		childrenInputSize = nC[0] > 0 ? inS[1] : 0;
		nCols = nC[0] > 0 ? inS[1] * nC[0] + inputSize : inputSize;
	}
};


class PC_Node_G : public IModule
{
public:
	PC_Node_G(PC_Node_G* n);

	~PC_Node_G() {};


	int nChildren;
	int inputSize, outputSize; // >= 1



	// Evolved parameters:

	// Structs containing the constant, evolved, matrix of parameters linking internal nodes.
	std::vector<InternalConnexion_G> toChildren; // nRows = children.inputSize
	InternalConnexion_G toOutput; // nRows = outputSize



	int getNParameters() override;

	PC_Node_G(std::ifstream& is);

	PC_Node_G(PC_Node_GFixedParameters& p);

	void save(std::ofstream& os) override;

	void mutate(float p) override;

	static PC_Node_G* combine(PC_Node_G** parents, float* weights, int nParents);
};

