#pragma once

#include <memory>

#include "InternalConnexion_G.h"
#include "config.h"


struct InternalConnexion_P {   // responsible of its pointers

	InternalConnexion_G* type;

	std::unique_ptr<float[]> H;
	std::unique_ptr<float[]> E;

	void randomInitH();

#ifdef DROPOUT
	void dropout();
#endif

	// Should not be called !
	// And strangely, is never called but removing its declaration causes an error.
	InternalConnexion_P(const InternalConnexion_P&) { __debugbreak();  type = nullptr; };
	
	// Should not be called !
	InternalConnexion_P() { __debugbreak();  type = nullptr; };

	InternalConnexion_P(InternalConnexion_G* type);

	void zeroE();

	~InternalConnexion_P() {};
};