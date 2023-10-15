#pragma once

#include <memory>
#include <iostream>

#include "InternalConnexion_G.h"
#include "MoEvCore.h"


struct InternalConnexion_P {   // responsible of its pointers

	InternalConnexion_G* type;

	std::unique_ptr<float[]> storage;

	std::vector<MMatrix> matrices;
	std::vector<MVector> vectors; 

	// LAYOUTS ARE DETAILED IN NODE_P::FORWARD()


#if !defined(PREDICTIVE_CODING)
#ifdef ABCD_ETA
	// modulation, one real value per line of H
	Eigen::VectorXf modulationV;
#endif


#ifdef SPRAWL_PRUNE
	// these arrays are used to store temporary results at each inference step.

	std::unique_ptr<float[]> tempBuffer1; // size nRows
	std::unique_ptr<float[]> tempBuffer2; // size nColumns

#endif
#endif

	void preTrialReset();
	

	// Should not be called !
	// And strangely, is never called but removing its declaration causes an error.
	InternalConnexion_P(const InternalConnexion_P&) { __debugbreak();  type = nullptr; };
	
	// Should not be called !
	InternalConnexion_P() { __debugbreak();  type = nullptr; };

	InternalConnexion_P(InternalConnexion_G* type);

	~InternalConnexion_P() { 
		//std::cerr << "CO DELETED !" << std::endl; 
	};
};