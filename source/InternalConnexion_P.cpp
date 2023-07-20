#include "InternalConnexion_P.h"

InternalConnexion_P::InternalConnexion_P(InternalConnexion_G* type) : type(type)
{
	int s = type->nLines * type->nColumns;

	H = std::make_unique<float[]>(s);
	E = std::make_unique<float[]>(s);

	randomInitH(); 

	// zeroE(); not necessary, because Node_P::preTrialReset() should be called before any computation. 
}

#ifdef DROPOUT
void InternalConnexion_P::dropout() {
	int s = type->nLines * type->nColumns;

	SET_BINOMIAL(s, .002f);
	int _nResets = BINOMIAL;
	for (int i = 0; i < _nResets; i++) {
		int id = INT_0X(s);


		H[id] = 0.0f;
		E[id] = 0.0f;

	}
}
#endif

void InternalConnexion_P::zeroE() {
	int s = type->nLines * type->nColumns;
	std::fill(E.get(), E.get() + s, 0.0f);
}


void InternalConnexion_P::randomInitH()
{
	float normalizator = .3f * powf((float)type->nColumns, -.5f); // Xavier or He ? No backprop so sticking with He
	int s = type->nLines * type->nColumns;

	for (int i = 0; i < s; i++) {

		H[i] = .2f * (UNIFORM_01 - .5f);
		//H[i] = NORMAL_01 * normalizator;

	}
}


