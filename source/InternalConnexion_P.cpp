#include "InternalConnexion_P.h"

InternalConnexion_P::InternalConnexion_P(InternalConnexion_G* type) : type(type)
{
	int s = type->nRows * type->nColumns;
	if (s == 0) return;

#ifdef SPRAWL_PRUNE
	tempBuffer1 = std::make_unique<float[]>(type->nRows);
	tempBuffer2 = std::make_unique<float[]>(type->nColumns);
#endif

	storage = std::make_unique<float[]>(s * N_DYNAMIC_MATRICES);

	float* _storagePtr = storage.get();

	matrices.resize(N_DYNAMIC_MATRICES);
	for (int i = 0; i < N_DYNAMIC_MATRICES; i++) {
		matrices[i] = _storagePtr;
		_storagePtr += s;
	}

	float normalizator = .3f * powf((float)type->nColumns, -.5f);
	for (int i = 0; i < s; i++) {

		//matrices[0][i] = normalizator * (float)(i%2);
		//matrices[0][i] = .2f * (UNIFORM_01 - .5f); // Risi Najarro
		//matrices[0][i] = NORMAL_01 * normalizator;

#ifdef ABCD_ETA
		matrices[0][i] = type->matricesR[3][i] + NORMAL_01 * normalizator; // lazy solution. Does not scale.
#else
		matrices[0][i] = NORMAL_01 * normalizator;
#endif
	}
	// zeroE(); not necessary, because Node_P::preTrialReset() should be called before any computation. 
}

void InternalConnexion_P::zeroE() {
	int s = type->nRows * type->nColumns;
	if (s == 0) return;

#ifdef ABCD_ETA
	std::fill(matrices[1], matrices[1] + s, 0.0f);
#elif defined(SPRAWL_PRUNE)
	std::fill(matrices[1], matrices[1] + s, 0.0f);
	std::fill(matrices[2], matrices[2] + s, 0.0f);
#endif
}



