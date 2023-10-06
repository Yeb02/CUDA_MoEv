#pragma once

#include "config.h"

#include <vector>
#include <fstream>

// The base virtual class which any trial should inherit from. 
// The score attribute must be a positive measure of the success of the run.
class Trial {
	
public:
	// Given the actions of the network, proceeds one step forward in the trial.
	virtual void step(const float* actions) = 0;

	// To be called at the end of the trial, AFTER fetching the score !
	// When sameSeed is true, the random values are kept between runs.
	virtual void reset(bool sameSeed = false) = 0;

	// copies the constant and per-run parameters of t. Must cast to derived class:
	// DerivedTrial* t = dynamic_cast<DerivedTrial*>(t0);
	virtual void copy(Trial* t0) = 0;

	// returns a pointer to a new instance OF THE DERIVED CLASS, cast to a pointer of the base class.
	virtual Trial* clone() = 0;

	// Handle for updates coming from the outer loop (main.cpp 's loop)
	virtual void outerLoopUpdate(void* data) = 0;

	std::vector<float> observations;

	// the required network dimensions
	int netInSize, netOutSize;

	float score;

	virtual ~Trial() = default; // otherwise derived destructors will not be called.

	bool isTrialOver;

protected:

	// the current elapsed steps in the trial. To be set to 0 in reset.
	int currentNStep;
};


class IModuleFixedParameters 
{
	// empty. Exists only to give a common base to parameters passed to ModulePopulations.
};


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

template <class Module>
class ModulePopulation;

class IAgent 
{
	
public:
	
	IAgent() 
	{
		nExperiencedTrials = 0;
		nInferencesOverLifetime = 0;
		lifetimeFitness = 0.0f;
		nModules = -1;
		modules = nullptr;
	}

	virtual ~IAgent() 
	{
		for (int i = 0; i < nModules; i++) modules[i]->nUsesInAgents--;
	};


	// Same as for the Module class. LAME
	// virtual Agent(std::ifstream& is) = 0; TODO find a way to do this...
	//virtual void createPhenotype(std::vector<ModulePopulation<class Module>*>& populations) = 0; An agent need not have a phenotype, in which case the overrides are empty

	virtual void save(std::ofstream& os) = 0;

	virtual float* getOutput() = 0;

	virtual void step(float* input) = 0;

	// Probably empty, but in rare cases it might be used.
	virtual void preTrialReset() = 0;

	virtual void destroyPhenotype() = 0;

	int nExperiencedTrials;

	float lifetimeFitness;

	int nModules;

	int nInferencesOverLifetime;

	std::unique_ptr<IModule* []> modules;
};