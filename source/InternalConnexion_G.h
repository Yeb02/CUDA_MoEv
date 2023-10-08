#pragma once

#include <memory>
#include <fstream>

#include <eigen-3.4.0/Eigen/Dense>
#include <eigen-3.4.0/Eigen/Core>

#include "Random.h"
#include "config.h"


typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MMatrix; // mapped matrix
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> MVector; // column mapped vector


inline float mutateDecayParam(float dp, float m = .15f);


struct InternalConnexion_G {

	static float decayParametersInitialValue;

	int nRows, nColumns;

	std::unique_ptr<float[]> storage;

	std::vector<MMatrix> matrices01;
	std::vector<MMatrix> matricesR;

	// vectors are of size nRows.
	std::vector<MVector> vectors01;
	std::vector<MVector> vectorsR;

	// LAYOUTS ARE DETAILED IN NODE_P::FORWARD()

	InternalConnexion_G()
	{ 
		nRows = -1; 
		nColumns = -1; 		
	};

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

	// internal util
	void createArraysFromStorage();
};
