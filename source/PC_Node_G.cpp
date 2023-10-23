#pragma once

#include "MoEvCore.h"
#ifdef PREDICTIVE_CODING

#include "PC_Node_G.h"

PC_Node_G::PC_Node_G(PC_Node_GFixedParameters& p) :
	IModule(),
	inputSize(p.inputSize), outputSize(p.outputSize), nChildren(p.nChildren),
	toChildren(), 
	toOutput(outputSize, p.nCols)
{
	toChildren.reserve(nChildren);
	for (int i = 0; i < nChildren; i++) {
		toChildren.emplace_back(p.childrenInputSize, p.nCols);
	}

};

PC_Node_G::PC_Node_G(PC_Node_G* n)
	: IModule()
{
	inputSize = n->inputSize;
	outputSize = n->outputSize;
	nChildren = n->nChildren;

	toChildren.reserve(n->toChildren.size());
	for (int i = 0; i < n->toChildren.size(); i++) toChildren.emplace_back(n->toChildren[i]);
	toOutput = n->toOutput;

}


int PC_Node_G::getNParameters()
{
	int cs = ((int)toChildren.size() == 0 ? 0 : toChildren[0].getNParameters());
#ifdef ACTIVATION_VARIANCE
	// TODO ?
#endif
	return cs + toOutput.getNParameters(); 
}


PC_Node_G* PC_Node_G::combine(PC_Node_G** parents, float* weights, int nParents)
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

	PC_Node_G* child = new PC_Node_G(parents[0]);

	InternalConnexion_G** connexions = new InternalConnexion_G * [nParents];


	// combines all connexions of the "connexions" array into childCo
	auto combineConnexions = [&](InternalConnexion_G* childCo)
	{
		for (int r = 0; r < childCo->nRows; r++)
		{
			for (int c = 0; c < childCo->nColumns; c++)
			{
				for (int j = 0; j < N_STATIC_MATRICES_01; j++)
				{
					childCo->matrices01[j](r,c) = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->matrices01[j](r,c);
				}
				for (int j = 0; j < N_STATIC_MATRICES_R; j++)
				{
					childCo->matricesR[j](r,c) = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->matricesR[j](r,c);
				}
			}
		}

		for (int i = 0; i < childCo->nRows; i++)
		{
			for (int j = 0; j < N_STATIC_VECTORS_01; j++)
			{
				childCo->vectors01[j](i) = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->vectors01[j](i);
			}
			for (int j = 0; j < N_STATIC_VECTORS_R; j++)
			{
				childCo->vectorsR[j](i) = connexions[proportionalParentPool[INT_0X(proportionalParentPoolSize)]]->vectorsR[j](i);
			}
		}
	};


	for (int i = 0; i < child->toChildren.size(); i++) {
		for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toChildren[i]; }
		combineConnexions(&child->toChildren[i]);
	}
	

	for (int j = 0; j < nParents; j++) { connexions[j] = &parents[j]->toOutput; }
	combineConnexions(&child->toOutput);

	delete[] connexions;

	return child;
}


PC_Node_G::PC_Node_G(std::ifstream& is) 
{

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);
	READ_4B(nChildren, is);
	
	// TODO to output, modulation.
	toChildren.reserve(nChildren);
	for (int i = 0; i < toChildren.size(); i++) toChildren.emplace_back(is);
}

void PC_Node_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);
	WRITE_4B(nChildren, os);

	// TODO to output, modulation.
	for (int i = 0; i < toChildren.size(); i++) toChildren[i].save(os);
	toOutput.save(os);
}

void PC_Node_G::mutate(float adjustedFMutationP)
{
	for (int i = 0; i < toChildren.size(); i++) toChildren[i].mutate(adjustedFMutationP);
	toOutput.mutate(adjustedFMutationP);
}

#endif
