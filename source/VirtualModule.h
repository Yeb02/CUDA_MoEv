#pragma once

#include "MoEvCore.h"

#include <vector>
#include <fstream>



class IModule
{
public:

	/*
	These 4 functions MUST be implemented by the derived classes, but there is just no way to specify
	an interface in c++ (like JAVA does). TODO . That's foul.

	With the derived class instead of Module :
	virtual Module(std::ifstream& is) = 0;
	virtual Module(IModuleFixedParameters* p) = 0;
	virtual Module(Module* m) = 0;
	virtual Module* combine(Module** parents, float* weights, int nParents) = 0;
	*/

	virtual void save(std::ofstream& os) = 0;


	// becomes false when it has been discarded from the population. "This" still exists because
	// it can be part of an alive Agent.
	bool isStillEvolved;


	// for use in its population
	float tempFitnessAccumulator;
	int nTempFitnessAccumulations;

	float lifetimeFitness;

	// how many times this module's population has called replaceModules. The call
	// during whixh this module was created does not count.
	int age;

	// How many occurences of this are in all alive agents. When it reaches 0 and  
	// isStillEvolved = false, "this" is deleted.
	int nUsesInAgents;

	virtual int getNParameters() = 0;

	virtual void mutate(float adjustedFMutationP) = 0;


	virtual ~IModule() {};

	IModule()
	{
		isStillEvolved = true;
		tempFitnessAccumulator = 0.0f;
		nTempFitnessAccumulations = 0;
		lifetimeFitness = 0.0f;
		nUsesInAgents = 0;
		age = 0;
	}

};
