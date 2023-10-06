#pragma once

#include <memory>
#include <fstream>

#include "Random.h"
#include "config.h"

inline float mutateDecayParam(float dp, float m = .15f);


struct InternalConnexion_G {

	static float decayParametersInitialValue;

	int nRows, nColumns;

	std::unique_ptr<float[]> storage;

	std::vector<float*> matrices01;
	std::vector<float*> matricesR;
	std::vector<float*> vectors01;
	std::vector<float*> vectorsR;

	InternalConnexion_G() { nRows = -1; nColumns = -1; };

	InternalConnexion_G(int nRows, int nColumns);

	InternalConnexion_G(const InternalConnexion_G& gc);

	InternalConnexion_G operator=(const InternalConnexion_G& gc);

	~InternalConnexion_G() {};

	InternalConnexion_G(std::ifstream& is);
	void save(std::ofstream& os);

	int getNParameters() {
		return nRows * nColumns * (N_STATIC_MATRICES_01 + N_STATIC_MATRICES_R)
			+ nRows * (N_STATIC_VECTORS_01 + N_STATIC_VECTORS_R);
	}

	void mutate(float p);
};
