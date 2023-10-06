#pragma once

#include <memory>
#include <cmath>
#include <string>
#include <thread>

#include "ModulePopulation.h"
#include "Network.h"
#include "Trial.h"
//#include "NoveltyEncoder.h"
#include "Random.h"


const enum SCORE_BATCH_TRANSFORMATION { NONE = 0, NORMALIZE = 1, RANK = 2};


// Contains the parameters for population evolution. They can be changed at each step.
struct SystemEvolutionParameters {

	// How many agents are alive at any given time. 
	int nAgents;

	// target fraction of the networks that is replaced at the end of a network cycle.
	float agentsReplacedFraction;

	// for all agents, lifetimeFitness(t) = lifetimeFitness(t-1) * accumulatedFitnessDecay + fitness(t)
	float accumulatedFitnessDecay;

	// how many trials each network is evaluated on before a call to replaceNetworks
	int nTrialsPerNetworkCycle;

	// how many calls are made to replaceNetworks before a call to replaceModules
	int nNetworksCyclesPerModuleCycle;



	// layer by layer number of evolved modules. Not an attribute of the system object
	int* nEvolvedModulesPerLayer;


	SystemEvolutionParameters() {};
};



class System {

public:	
	~System() 
	{
		stopThreads();
		for (int i = 0; i < nAgents; i++) {
			delete agents[i];
		}
		delete[] agents;
	};

	void evolve(int nSteps);


	System(Trial** trials, SystemEvolutionParameters& sParams, NetworkParameters& nParams, ModulePopulationParameters& mpParams, int nThreads);


	void setEvolutionParameters(SystemEvolutionParameters params) 
	{
		this->nAgents = params.nAgents;
		this->agentsReplacedFraction = params.agentsReplacedFraction;

		this->accumulatedFitnessDecay = params.accumulatedFitnessDecay;

		this->nTrialsPerNetworkCycle = params.nTrialsPerNetworkCycle;
		this->nNetworksCyclesPerModuleCycle = params.nNetworksCyclesPerModuleCycle;
	}

	
	void saveFittestSpecimen();


	void startThreads();


	void stopThreads();


private:
	// 1 per thread
	Trial** trials;

	//std::unique_ptr<NoveltyEncoder> noveltyEncoder;


	std::vector<ModulePopulation<Node_G>*> populations;

	// Layer by layer total number of modules in a network. Computed once as a util.
	std::vector<int> nModulesPerNetworkLayer;

	IAgent** agents;

	// Util for agents.
	int nModulesPerAgent;


	// lifetime fitness threshold for networks. Dynamic quantity, adjusted after each replacement step
	// to match the actual replaced fraction with networksReplacedFraction more closely at the next step.
	float currentAgentReplacementTreshold;

	// To save progress regularly
	int fittestSpecimen;



	void accumulateModuleFitnesses();

	void replaceNetworks();

	void replaceModules();

	void zeroModulesAccumulators();

	
	// How threading goes here: Each thread is assigned a chunk of the agents array, and a trial.
	// At each module cycle, the threads evaluate their agents on their trial.

	int nThreads;
	int nAgentsPerThread; // = nAgents/nThreads
	std::vector<std::thread> threads;
	int threadIteration; // util
	bool mustTerminate;
	void perThreadMainLoop(const int threadID);


	// EVOLUTION PARAMETERS: 
	// Set with a SystemEvolutionParameters struct. Description in the struct definition.
	int nAgents;
    float agentsReplacedFraction;
    float accumulatedFitnessDecay; 
	int nTrialsPerNetworkCycle;
	int nNetworksCyclesPerModuleCycle;
};