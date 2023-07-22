#include "Node_G.h"

Node_G::Node_G(int* inS, int* outS, int nC) :
	inputSize(inS[0]), outputSize(outS[0]), nChildren(nC),
	toModulation(MODULATION_VECTOR_SIZE, computeNCols(inS, outS, nC)),
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

	toChildren = n->toChildren;
	toModulation = n->toModulation;
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
	Node_G* child = new Node_G(parents[0]);

	InternalConnexion_G** connexions = new InternalConnexion_G * [nParents];


	// copies parentCo's  matrices id-th components into childCo's matrices
	auto copyMatComponents = [](InternalConnexion_G* parentCo, InternalConnexion_G* childCo, int id)
	{
		childCo->A[id] = parentCo->A[id];
		childCo->B[id] = parentCo->B[id];
		childCo->C[id] = parentCo->C[id];
		childCo->eta[id] = parentCo->eta[id];
	};

	// copies parentCo's  arrays id-th components into childCo's array
	auto copyArrComponents = [](InternalConnexion_G* parentCo, InternalConnexion_G* childCo, int id)
	{
		childCo->kappa[id] = parentCo->kappa[id];

#ifdef STDP
		childCo->STDP_mu[id] = parentCo->STDP_mu[id];
		childCo->STDP_lambda[id] = parentCo->STDP_lambda[id];
#endif
	};

	// combines all connexions of the "connexions" array into childCo
	auto combineConnexions = [connexions, nParents, &copyArrComponents, &copyMatComponents](InternalConnexion_G* childCo)
	{
		int sMat = childCo->nLines * childCo->nColumns;
		for (int i = 0; i < sMat; i++)
		{
			copyMatComponents(childCo, connexions[INT_0X(nParents)], i);
		}

		int sArr = childCo->nLines;
		for (int i = 0; i < sArr; i++)
		{
			copyArrComponents(childCo, connexions[INT_0X(nParents)], i);
		}
	};


	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toChildren; }
	combineConnexions(&child->toChildren);

	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toOutput; }
	combineConnexions(&child->toOutput);

	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toModulation; }
	combineConnexions(&child->toModulation);
	

	delete[] connexions;

	return child;
}

Node_G::Node_G(std::ifstream& is) {

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);
	
	toChildren = InternalConnexion_G(is);
	toModulation = InternalConnexion_G(is);
	toOutput = InternalConnexion_G(is);

	isInModuleArray = false;
}

void Node_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	
	toChildren.save(os);
	toModulation.save(os);
	toOutput.save(os);
}

void Node_G::mutateFloats(float adjustedFMutationP)
{
	toChildren.mutateFloats(adjustedFMutationP);
	toModulation.mutateFloats(adjustedFMutationP);
	toOutput.mutateFloats(adjustedFMutationP);
}


