#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

#endif



#include "Population.h"
#include "Random.h"

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#pragma comment(lib, "cublas.lib") // Not linked by default in the cuda template. 

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;



int main()
{
    LOG("Seed : " << seed);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

#ifdef ROCKET_SIM_T
    // Path to where you dumped rocket league collision meshes.
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif

    int channelSize = 2;
    int nAgentsPerGroup = 3;

#ifdef NO_GROUP
    nAgentsPerGroup = 1; // DO NOT EDIT
#endif

    Trial* innerTrial;

    // construct innerTrial.
    {
#ifdef CARTPOLE_T
        innerTrial = new CartPoleTrial(true); // bool : continuous control.
#elif defined XOR_T
        innerTrial = new XorTrial(4, 5);  // int : vSize, int : delay
#elif defined TMAZE_T
        innerTrial = new TMazeTrial(false);
#elif defined N_LINKS_PENDULUM_T
        innerTrial = new NLinksPendulumTrial(false, 2);
#elif defined MEMORY_T
        innerTrial = new MemoryTrial(2, 5, 5, true); // int nMotifs, int motifSize, int responseSize, bool binary = true
#elif defined ROCKET_SIM_T
        innerTrial = new RocketSimTrial();
#endif
    }

    GroupTrial groupTrial(innerTrial, nAgentsPerGroup, channelSize);

    int trialObservationsSize = groupTrial.netInSize;
    int trialActionsSize = groupTrial.netOutSize;    
#ifdef NO_GROUP 
    trialObservationsSize = groupTrial.innerTrial->netInSize;   
    trialActionsSize = groupTrial.innerTrial->netOutSize;       
#endif

    int inputInterfaceSize = 4;
    int outputInterfaceSize = 4;

    const int inputMLPnLayers = 2;  // >= 1
    const int outputMLPnLayers = 2; // >= 1

    if (inputMLPnLayers == 0) inputInterfaceSize = trialObservationsSize;
    if (outputMLPnLayers == 0) outputInterfaceSize = trialActionsSize;

    int inputMLPsizes[inputMLPnLayers+1] = { trialObservationsSize, 5, inputInterfaceSize };
    int outputMLPsizes[outputMLPnLayers+1] = { outputInterfaceSize, 3, trialActionsSize };

    /* 
    // A structurally non trivial example for debugging
    const int nLayers = 3;
    int inSizes[nLayers] = { inputInterfaceSize, 8, 4};
    int outSizes[nLayers] = { outputInterfaceSize, 7, 3};
    int nChildrenPerLayer[nLayers] = {2, 1, 0};
    int nEvolvedModulesPerLayer[nLayers] = {64, 128, 128};
    float moduleReplacedFractions[nLayers] = {.3f, .3f, .3f};
    */

    const int nLayers = 1;
    int inSizes[nLayers] = { inputInterfaceSize };
    int outSizes[nLayers] = { outputInterfaceSize };
    int nChildrenPerLayer[nLayers] = { 0 }; // Must end with 0
    int nEvolvedModulesPerLayer[nLayers] = { 64 };
    float moduleReplacedFractions[nLayers] = { .3f }; // must be in [0,.5]


    InternalConnexion_G::decayParametersInitialValue = .3f;

    PopulationEvolutionParameters params;

    params.nSpecimens = 128;
    params.nTrialsPerGroup = 2;
    params.maxNParents = 10;
    params.nLayers = nLayers;
    params.inSizes = inSizes;
    params.outSizes = outSizes;
    params.nChildrenPerLayer = nChildrenPerLayer;
    params.nEvolvedModulesPerLayer = nEvolvedModulesPerLayer;
    params.moduleReplacedFractions = moduleReplacedFractions; 
    params.networkReplacedFraction = .2f; //in [0,.5]
    params.voteValue = .3f; // > 0
    params.accumulatedFitnessDecay = .9f; //in [0,1]
    params.baseMutationProbability = 1.0f;//in [0,1]
    params.consanguinityDistance = 3;  // MUST BE >= 1

    params.useInMLP = false;  // If false, all parameters regarding in  MLPs have no incidence
    params.useOutMLP = false; // If false, all parameters regarding out MLPs have no incidence
    params.inputMLPnLayers = inputMLPnLayers;  
    params.outputMLPnLayers = outputMLPnLayers;  
    params.inMLPReplacedFraction = .3f; // must be in [0,.5]
    params.outMLPReplacedFraction = .3f;// must be in [0,.5]
    params.inputMLPsizes = inputMLPsizes;  
    params.outputMLPsizes = outputMLPsizes;  
    params.nInMLPs = 64;
    params.nOutMLPs = 64;


    int nSteps = 10000;

    
    Population population(&groupTrial, params);

    population.evolve(nSteps);

    return 0;
}
