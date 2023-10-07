#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>

#include "VirtualClasses.h"
#include "Random.h"
#include "Config.h"
#include "InternalConnexion_G.h"



// Util:
inline int binarySearch(std::vector<float>& proba, float value) {
	int inf = 0;
	int sup = (int)proba.size() - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

inline int binarySearch(float* proba, float value, int size) {
	int inf = 0;
	int sup = size - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}


struct Node_GFixedParameters : public IModuleFixedParameters
{
	int nChildren;
	int inputSize, outputSize;

	int nCols;
	int toChildrenNLines;

	Node_GFixedParameters(int* inS, int* outS, int* nC) :
		nChildren(nC[0]), inputSize(inS[0]), outputSize(outS[0])
	{

		toChildrenNLines = nC[0] > 0 ? outS[1] * nC[0] : 0;
		nCols = nC[0] > 0 ? inS[1] * nC[0] + inputSize : inputSize;
	}
};


class Node_G : public IModule
{
public:
	Node_G(Node_G* n);

	~Node_G() {};


	int nChildren;
	int inputSize, outputSize; // >= 1

	// Structs containing the constant, evolved, matrix of parameters linking internal nodes.
	// The name specifies the type of node that takes the result of the matrix operations as inputs.
	// nColumns = this.inputSize + MODULATION_VECTOR_SIZE + sum(children.inputSize)
	InternalConnexion_G toChildren; // nRows = sum(children.inputSize) 
	InternalConnexion_G toOutput; // nRows = outputSize


	int getNParameters() override {return toChildren.getNParameters() + toOutput.getNParameters();}

	Node_G(std::ifstream& is);

	Node_G(Node_GFixedParameters& p);

	void save(std::ofstream& os) override;

	void mutate(float p) override;

	static Node_G* combine(Node_G** parents, float* weights, int nParents);
};

