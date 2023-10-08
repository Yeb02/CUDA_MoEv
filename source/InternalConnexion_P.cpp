#include "InternalConnexion_P.h"

InternalConnexion_P::InternalConnexion_P(InternalConnexion_G* _type) : 
	type(_type)
{
	int s = type->nRows * type->nColumns;
	if (s == 0) return; // happens when this is a toChildren connexion and there are 0 children.

#ifdef SPRAWL_PRUNE
	tempBuffer1 = std::make_unique<float[]>(type->nRows);
	tempBuffer2 = std::make_unique<float[]>(type->nColumns);
#endif

#ifdef ABCD_ETA
	modulationV.resize(type->nRows);
#endif

	storage = std::make_unique<float[]>(s * N_DYNAMIC_MATRICES + type->nRows * N_DYNAMIC_VECTORS);

	float* _storagePtr = storage.get();


	matrices.reserve(N_DYNAMIC_MATRICES);
	for (int i = 0; i < N_DYNAMIC_MATRICES; i++) {
		matrices.emplace_back(_storagePtr, type->nRows, type->nColumns);
		_storagePtr += s;
	}

	vectors.reserve(N_DYNAMIC_VECTORS);
	for (int i = 0; i < N_DYNAMIC_VECTORS; i++) {
		vectors.emplace_back(_storagePtr, type->nRows);
		_storagePtr += type->nRows;
	}

	// initializations:
	float normalizator = .3f * powf((float)type->nColumns, -.5f);
	for (int i = 0; i < type->nRows; i++) {
		for (int j = 0; j < type->nColumns; j++) {
			//matrices[0](i, j) = normalizator * (float)(i%2);
			//matrices[0](i, j) = .2f * (UNIFORM_01 - .5f); // Risi Najarro
			//matrices[0](i, j) = NORMAL_01 * normalizator;

#ifdef ABCD_ETA
			matrices[0](i, j) = type->matricesR[3](i, j) + NORMAL_01 * normalizator; // lazy solution. Does not scale.
#else	
			matrices[0](i, j) = NORMAL_01 * normalizator;
#endif
		}
	}

	// zeroE(); not necessary, because Node_P::preTrialReset() should be called before any computation. 
}

void InternalConnexion_P::zeroE() {
	int s = type->nRows * type->nColumns;
	if (s == 0) return;

#ifdef ABCD_ETA
	std::fill(matrices[1].data(), matrices[1].data() + s, 0.0f);
#elif defined(SPRAWL_PRUNE)
	std::fill(matrices[1].data(), matrices[1].data() + s, 0.0f);
	std::fill(matrices[2].data(), matrices[2].data() + s, 0.0f);
#endif
}



