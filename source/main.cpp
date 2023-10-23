#pragma once

#include <iostream>


#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);
#endif


#include <eigen-3.4.0/Eigen/Core>

#include "System.h"
#include "MoEvCore.h"
//#include "MNIST.h"

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


using namespace std;

// The .h in which the struct is defined does not have a .cpp because the main class is a template...
int PhylogeneticNode::maxPhylogeneticDepth = 0;


int main()
{
    // https://docs.huihoo.com/eigen/3/TopicMultiThreading.html
    Eigen::initParallel();

    LOG("Seed : " << seed);


#ifdef ROCKET_SIM_T
    // Path to where you dumped rocket league collision meshes.
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif

    int nThreads = std::thread::hardware_concurrency();
#ifdef _DEBUG
    nThreads = 1; // Because multi-threaded functions are difficult debug line by line in VS
#endif
    LOG(nThreads << " concurrent threads are supported at hardware level.");

    std::vector<Trial*> trials;
    trials.resize(nThreads);


    for (int i = 0; i < nThreads; i++) {
#ifdef CARTPOLE_T
        trials[i] = new CartPoleTrial(true); // bool : continuous control.
#elif defined XOR_T
        trials[i] = new XorTrial(4, 5);  // int : vSize, int : delay
#elif defined TEACHING_T
        trials[i] = new TeachingTrial(5, 5);
#elif defined TMAZE_T
        trials[i] = new TMazeTrial(false);
#elif defined N_LINKS_PENDULUM_T
        trials[i] = new NLinksPendulumTrial(false, 2);
#elif defined MEMORY_T
        trials[i] = new MemoryTrial(2, 5, 5, true); // int nMotifs, int motifSize, int responseSize, bool binary = true
#elif defined ROCKET_SIM_T
        trials[i] = new RocketSimTrial();
#endif
    }

 
    int trialObservationsSize = trials[0]->netInSize;
    int trialActionsSize = trials[0]->netOutSize;

    
#define TRIVIAL_ARCHITECTURE
#ifdef TRIVIAL_ARCHITECTURE
    const int nLayers = 1;
    int inSizes[nLayers] = { trialObservationsSize };
    int outSizes[nLayers] = { trialActionsSize };
    int nChildrenPerLayer[nLayers] = { 0 }; // Must end with 0
    int nEvolvedModulesPerLayer[nLayers] = { 64 };
#else
    // A structurally non trivial example
    const int nLayers = 2;
    int inSizes[nLayers] = { trialObservationsSize, 4};
    int outSizes[nLayers] = { trialActionsSize, 3};
    int nChildrenPerLayer[nLayers] = { 2, 0 }; // Must end with 0
    int nEvolvedModulesPerLayer[nLayers] = { 32, 64 };
#endif
    
#ifdef ACTION_L_OBS_O
    {
        int temp = inSizes[0];
        inSizes[0] = outSizes[0];
        outSizes[0] = temp;
    }
#endif


    InternalConnexion_G::decayParametersInitialValue = .5f;


    SystemEvolutionParameters sParams;

    sParams.nAgents = 64;
    sParams.agentsReplacedFraction = .2f; //in [0,.5]
    sParams.nEvolvedModulesPerLayer = nEvolvedModulesPerLayer;
    sParams.nEvaluationTrialsPerAgentCycle = 4;
    int nSupervisedTrialsPerAgentCycle = 4; // either 0, or >= nThreads !! The logic next line enforces it.
    sParams.nSupervisedTrialsPerAgentCycle = (nSupervisedTrialsPerAgentCycle == 0 ? 0 : std::max(nSupervisedTrialsPerAgentCycle, nThreads));
    sParams.nAgentCyclesPerModuleCycle = 2;
    sParams.scoreTransformation = RANK;
    sParams.accumulatedFitnessDecay = .8f;


    AGENT_PARAMETERS aParams;

    aParams.nLayers = nLayers;
    aParams.inSizes = inSizes;
    aParams.outSizes = outSizes;
    aParams.nChildrenPerLayer = nChildrenPerLayer;


    // All parameters excepted maxPhylogeneticDepth could be per modulePopulation, but to limit the number of
    // hyperparameters only nModules is specific to each population (and therefore set by the system for simplicity).
    ModulePopulationParameters mpParams;

    //mpParams.nModules = sParams.nEvolvedModulesPerLayer[l]; 
    mpParams.accumulatedFitnessDecay = .8f; //in [0,1]
    mpParams.maxNParents = 10;
    mpParams.maxPhylogeneticDepth = 10;
    mpParams.moduleReplacedFraction = .3f; // must be in [0,.5]
    mpParams.moduleElitePercentile = .3f; // must be in [0,.5]
    mpParams.baseMutationProbability = .5f; // in [0,1]
    mpParams.consanguinityDistance = 1; // must be >= 1


    int nSteps = 10000;


    System system(trials.data(), sParams, aParams, mpParams, nThreads);

    system.evolve(nSteps);

    return 0;
}
