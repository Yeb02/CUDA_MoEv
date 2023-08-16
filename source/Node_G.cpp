#include "Node_G.h"

Node_G::Node_G(int* inS, int* outS, int nC) :
	inputSize(inS[0]), outputSize(outS[0]), nChildren(nC),
	toChildren(nC > 0 ? nC * inS[1] : 0, computeNCols(inS, outS, nC)),
	toOutput(outS[0], computeNCols(inS, outS, nC))
{
	isInModuleArray = false;
	tempFitnessAccumulator = 0.0f;
	nTempFitnessAccumulations = 0;
	lifetimeFitness = 0.0f;
	nUsesInNetworks = 0;
};

Node_G::Node_G(Node_G* n) {

	inputSize = n->inputSize;
	outputSize = n->outputSize;
	nChildren = n->nChildren;

	toChildren = n->toChildren;
	toOutput = n->toOutput;

	isInModuleArray = false;
	tempFitnessAccumulator = 0.0f;
	nTempFitnessAccumulations = 0;
	lifetimeFitness = 0.0f;
	nUsesInNetworks = 0;
}

// Sparse version. TODO continuous (GPU enabled ?)
Node_G* Node_G::combine(Node_G** parents, float* weights, int nParents)
{
	const int proportionalParentPoolSize = 10 * 10; // TODO  Should be resolution * maxNParents (population's
	int proportionalParentPool[proportionalParentPoolSize]; 

	float invWSum = 0.0f;
	for (int i = 0; i < nParents; i++) 
	{
		invWSum += weights[i];
	}
	invWSum = 1.0f / invWSum;
	int id = 0;
	for (int i = 0; i < nParents; i++)
	{
		int* i0 = proportionalParentPool + id;
		int stride = (int)((float)proportionalParentPoolSize * invWSum * weights[i]);
		std::fill(i0, i0 + stride, i);
		id += stride;
	}
	std::fill(proportionalParentPool+id, proportionalParentPool+ proportionalParentPoolSize, 0);

	Node_G* child = new Node_G(parents[0]);

	InternalConnexion_G** connexions = new InternalConnexion_G * [nParents];


	// combines all connexions of the "connexions" array into childCo
	auto combineConnexions = [&](InternalConnexion_G* childCo)
	{
		int sMat = childCo->nRows * childCo->nColumns;
		for (int i = 0; i < sMat; i++)
		{
			for (int j = 0; j < N_STATIC_MATRICES_01; j++) 
			{
				childCo->matrices01[j][i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->matrices01[j][i];
			}
			for (int j = 0; j < N_STATIC_MATRICES_R; j++)
			{
				childCo->matricesR[j][i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->matricesR[j][i];
			}
		}

		int sArr = childCo->nRows;
		for (int i = 0; i < sArr; i++)
		{
			for (int j = 0; j < N_STATIC_VECTORS_01; j++)
			{
				childCo->vectors01[j][i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->vectors01[j][i];
			}
			for (int j = 0; j < N_STATIC_VECTORS_R; j++)
			{
				childCo->vectorsR[j][i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->vectorsR[j][i];
			}
		}
	};


	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toChildren; }
	combineConnexions(&child->toChildren);

	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toOutput; }
	combineConnexions(&child->toOutput);
	

	delete[] connexions;

	return child;
}

Node_G::Node_G(std::ifstream& is) {

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);
	
	toChildren = InternalConnexion_G(is);
	toOutput = InternalConnexion_G(is);

	isInModuleArray = false;
}

void Node_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	
	toChildren.save(os);
	toOutput.save(os);
}

void Node_G::mutateFloats(float adjustedFMutationP)
{
	toChildren.mutateFloats(adjustedFMutationP);
	toOutput.mutateFloats(adjustedFMutationP);
}


