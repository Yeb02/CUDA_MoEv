#pragma once

#include "MLP_G.h"


MLP_G::MLP_G(int* ls, int nl) 
{
	nLayers = nl;
	sizes = ls;


	isStillEvolved = false;
	tempFitnessAccumulator = 0.0f;
	nTempFitnessAccumulations = 0;
	lifetimeFitness = 0.0f;
	nUsesInNetworks = 0;


	for (int i = 0; i < nLayers; i++)
	{
		float f = powf((float)sizes[i], -.5f);
		int sW = sizes[i] * sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = NORMAL_01 * f;
		}

		int sB = sizes[i + 1];
		Bs.emplace_back(new float[sizes[i + 1]]);
		for (int j = 0; j < sizes[i + 1]; j++) {
			//B0s[i][j] = NORMAL_01;
			Bs[i][j] = 0.0f;
		}
	}
}


MLP_G::MLP_G(MLP_G* pn) 
{
	nLayers = pn->nLayers;
	sizes = pn->sizes;


	isStillEvolved = false;
	tempFitnessAccumulator = 0.0f;
	nTempFitnessAccumulations = 0;
	lifetimeFitness = 0.0f;
	nUsesInNetworks = 0;

	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		std::copy(pn->Ws[i].get(), pn->Ws[i].get() + sW, Ws[i].get());

		int sB = sizes[i + 1];
		Bs.emplace_back(new float[sizes[i + 1]]);
		std::copy(pn->Bs[i].get(), pn->Bs[i].get() + sB, Bs[i].get());
	}
}


void MLP_G::mutate(float p)
{
	auto mutateMatrix = [](float* matrix, int matSize, float f, float regularizer)
	{
		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(matSize);

			matrix[matrixID] *= regularizer + NORMAL_01 * f;
			matrix[matrixID] += NORMAL_01 * f;
		}
	};


	for (int i = 0; i < nLayers; i++)
	{
		float f = .5f*powf((float)sizes[i], -.5f);

		int sW = sizes[i] * sizes[i + 1];
		SET_BINOMIAL(sW, p);
		mutateMatrix(Ws[i].get(), sW, f, .95f);

		int sB = sizes[i + 1];
		SET_BINOMIAL(sB, p);
		mutateMatrix(Bs[i].get(), sB, .2f, .9f);
	}
}


MLP_G* MLP_G::combine(MLP_G** parents, float* weights, int nParents)
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
	std::fill(proportionalParentPool + id, proportionalParentPool + proportionalParentPoolSize, 0);

	MLP_G* child = new MLP_G(parents[0]);

	float** arrays = new float*[nParents];



	for (int i = 0; i < child->nLayers; i++) 
	{
		for (int j = 0; j < nParents; j++)
		{
			arrays[j] = parents[j]->Ws[i].get();
		}
		int sW = child->sizes[i] * child->sizes[i + 1];
		for (int j = 0; j < sW; j++)
		{
			child->Ws[i][j] = arrays[proportionalParentPool[INT_0X(proportionalParentPoolSize)]][j];
		}

		for (int j = 0; j < nParents; j++)
		{
			arrays[j] = parents[j]->Bs[i].get();
		}
		int sB = child->sizes[i + 1];
		for (int j = 0; j < sB; j++)
		{
			child->Bs[i][j] = arrays[proportionalParentPool[INT_0X(proportionalParentPoolSize)]][j];
		}
	}
	
	delete[] arrays;

	return child;
}
