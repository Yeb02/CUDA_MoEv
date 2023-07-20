#pragma once

#include <memory>
#include <fstream>

#define WRITE_4B(i, os) os.write(reinterpret_cast<const char*>(&i), 4);
#define READ_4B(i, is) is.read(reinterpret_cast<char*>(&i), 4);

#include "Random.h"
#include "config.h"

inline float mutateDecayParam(float dp, float m = .15f);


struct InternalConnexion_G {


	int nLines, nColumns;

	std::unique_ptr<float[]> A;
	std::unique_ptr<float[]> B;
	std::unique_ptr<float[]> C;
	std::unique_ptr<float[]> eta;	// in [0, 1]

#ifdef STDP
	std::unique_ptr<float[]> STDP_mu; // in [0, 1]
	std::unique_ptr<float[]> STDP_lambda;// in [0, 1]
#endif

	std::unique_ptr<float[]> kappa;// in [0, 1]


	InternalConnexion_G() { nLines = -1; nColumns = -1; };

	InternalConnexion_G(int nLines, int nColumns);

	InternalConnexion_G(const InternalConnexion_G& gc);

	InternalConnexion_G operator=(const InternalConnexion_G& gc);

	~InternalConnexion_G() {};

	InternalConnexion_G(std::ifstream& is);
	void save(std::ofstream& os);

	int getNParameters() {
		return 4 * nLines * nColumns; // not counting parameters used only nLines times
	}

	void mutateFloats(float p);
};
