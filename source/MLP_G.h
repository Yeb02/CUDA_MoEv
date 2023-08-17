#pragma once

#include "config.h"
#include "Random.h"

#include <memory>
#include <vector>



struct MLP_G 
{

	// = n hidden layers + 1
	int nLayers;

	// size nLayers
	std::vector<std::unique_ptr<float[]>> Ws;

	// size nLayers
	std::vector<std::unique_ptr<float[]>> Bs;

	// the size of each layer. Size nLayers + 1 !
	int* sizes;

	// Utils for network:
	bool isStillEvolved;
	float tempFitnessAccumulator;
	int nTempFitnessAccumulations;
	float lifetimeFitness;
	int nUsesInNetworks;

	int getNParameters() 
	{
		int np = 0;
		for (int i = 0; i < nLayers; i++) {
			np += (sizes[i] + 1) * sizes[i + 1];
		}
		return np;
	}

	static MLP_G* combine(MLP_G** parents, float* weights, int nParents);

	void mutate(float p);

	MLP_G(int* ls, int nl);
	MLP_G(MLP_G* pn);

	~MLP_G() {};
};