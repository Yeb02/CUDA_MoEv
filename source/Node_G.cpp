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
	nChildren = n->nChildren;

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
		int sMat = childCo->nLines * childCo->nColumns;
		for (int i = 0; i < sMat; i++)
		{
			//childCo->A[id] = connexions[INT_0X(nParents)]->A[id];

			childCo->A[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->A[i];
			childCo->B[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->B[i];
			childCo->C[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->C[i];
			childCo->eta[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->eta[i];
		}

		int sArr = childCo->nLines;
		for (int i = 0; i < sArr; i++)
		{
			childCo->kappa[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->kappa[i];

#ifdef STDP
			childCo->STDP_mu[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->STDP_mu[i];
			childCo->STDP_lambda[i] = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->STDP_lambda[i];
#endif
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


