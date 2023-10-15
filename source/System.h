#pragma once

#include <memory>
#include <cmath>
#include <string>
#include <thread>

#include "MoEvCore.h"
#include "ModulePopulation.h"
#include "Trial.h"

#ifdef PREDICTIVE_CODING
#include "PC_Network.h"
#else
#include "HebbianNetwork.h"
#endif

const enum SCORE_BATCH_TRANSFORMATION { NONE = 0, NORMALIZE = 1, RANK = 2};


// Contains the parameters for population evolution. They can be changed at each step.
struct SystemEvolutionParameters {

	// How many agents are alive at any given time. 
	int nAgents;

	// target fraction of the networks that is replaced at the end of a network cycle.
	float agentsReplacedFraction;

	// for all agents, lifetimeFitness(t) = lifetimeFitness(t-1) * accumulatedFitnessDecay + 
	// fitness(t) *(1-accumulatedFitnessDecay)
	float accumulatedFitnessDecay;

	// how many trials each network is evaluated on before a call to replaceHebbianNetworks
	int nTrialsPerAgentCycle;

	// how many calls are made to replaceHebbianNetworks before a call to replaceModules
	int nAgentCyclesPerModuleCycle;

	// Raw scores can be transformed into a more sensical measure of the fitness of agents.
	// Either NONE, RANKING or NORMALIZATION. In doubt, use RANKING. NONE only if you know 
	// what you are doing.
	SCORE_BATCH_TRANSFORMATION scoreTransformation;



	// layer by layer number of evolved modules. Not an attribute of the system object
	int* nEvolvedModulesPerLayer;


	SystemEvolutionParameters() {};
};



class System {

public:	
	~System();


	void evolve(int nSteps);


	System(Trial** trials, SystemEvolutionParameters& sParams, AGENT_PARAMETERS& aParams, ModulePopulationParameters& mpParams, int nThreads);


	void setEvolutionParameters(SystemEvolutionParameters params) 
	{
		this->nAgents = params.nAgents;
		this->agentsReplacedFraction = params.agentsReplacedFraction;

		this->accumulatedFitnessDecay = params.accumulatedFitnessDecay;

		this->nTrialsPerAgentCycle = params.nTrialsPerAgentCycle;
		this->nAgentCyclesPerModuleCycle = params.nAgentCyclesPerModuleCycle;

		this->scoreTransformation = params.scoreTransformation;
	}

	
	// saves genotype (and phenotype if it exists) of the current best agent.
	void saveBestAgent();

	// literally. To be used to pause a run, and potentially transfer it to another machine.
	void saveEverything();


	void startThreads();


	void stopThreads();


private:
	// 1 per thread. TODO "this" can optionnally store a list of nTrialsPerAgentCycle trial states
	// so that all agents are evaluated exactly on the same trials.
	Trial** trials;

	// Holds the scores per trial per agent, potentially transformed by a normalization or ranking operation.
	std::vector<float*> agentsScores;


	std::vector<ModulePopulation*> populations;

	AGENT** agents;

	// Util for agents.
	int nModulesPerAgent;


	// lifetime fitness threshold for networks. Dynamic quantity, adjusted after each replacement step
	// to match the actual replaced fraction with networksReplacedFraction more closely at the next step.
	float currentAgentReplacementTreshold;

	// To save progress regularly
	int fittestSpecimen;


	void replaceAgents();

	
	// How threading goes here: Each thread is assigned a chunk of the agents array, and a trial.
	// At each module cycle, the threads evaluate their agents on their trial.

	int nThreads;
	int nAgentsPerThread; // = nAgents/nThreads
	std::vector<std::thread> threads;
	int threadIteration; // util
	bool mustTerminate;
	void perThreadMainLoop(const int threadID);

	// prints out fitness information.
	void log();

	// EVOLUTION PARAMETERS: 
	// Set with a SystemEvolutionParameters struct. Description in the struct definition.
	int nAgents;
    float agentsReplacedFraction;
    float accumulatedFitnessDecay; 
	int nTrialsPerAgentCycle;
	int nAgentCyclesPerModuleCycle;
	SCORE_BATCH_TRANSFORMATION scoreTransformation;
};