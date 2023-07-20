#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <tuple>
#include <fstream>

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

struct Node_G {

	Node_G(int* inS, int* outS, int nC, Node_G* c);

	Node_G(Node_G* n);

	~Node_G() {};

	Node_G(std::ifstream& is);
	void save(std::ofstream& os);

	// stupid but thats the only way to do complex operations in an initializer list.
	static int computeNCols(int* inS, int* outS, int nC) {
		int cIn = nC > 0 ? outS[1] * nC : 0;
		return inS[0] + MODULATION_VECTOR_SIZE + cIn;
	}

	int inputSize, outputSize; // >= 1

	
	// Contains pointers to the children. The same child can appear multiple times.
	std::vector<Node_G*> children;

	// Structs containing the constant, evolved, matrix of parameters linking internal nodes.
	// The name specifies the type of node that takes the result of the matrix operations as inputs.
	// nColumns = this.inputSize + MODULATION_VECTOR_SIZE + sum(complexChild.inputSize) + sum(memoryChild.inputSize).
	InternalConnexion_G toChildren; // nLines = sum(children.inputSize) 
	InternalConnexion_G toModulation; // nLines = MODULATION_VECTOR_SIZE
	InternalConnexion_G toOutput; // nLines = outputSize


	// returns the number of evolved floating point parameters.
	int getNParameters() 
	{
		return  toChildren.getNParameters() + toModulation.getNParameters() + toOutput.getNParameters();
	}

	// Mutate real-valued floating point parameters.
	void mutateFloats(float adjustedFMutationP);

};

