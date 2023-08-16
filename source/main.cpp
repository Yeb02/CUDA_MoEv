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

 
    /*const int nLayers = 3;
    int inSizes[nLayers] = { groupTrial.netInSize, 8, 4};
    int outSizes[nLayers] = { groupTrial.netOutSize, 7, 3};
    int nChildrenPerLayer[nLayers] = {2, 1, 0};
    int nEvolvedModulesPerLayer[nLayers] = {64, 128, 128};
    float moduleReplacedFractions[nLayers] = {.3f, .3f, .3f};*/

    const int nLayers = 1;
    int inSizes[nLayers] = { groupTrial.netInSize};
    int outSizes[nLayers] = { groupTrial.netOutSize};
    int nChildrenPerLayer[nLayers] = { 0 };
    int nEvolvedModulesPerLayer[nLayers] = { 64 };
    float moduleReplacedFractions[nLayers] = { .3f };

#ifdef NO_GROUP 
    inSizes[0] = groupTrial.innerTrial->netInSize;   // DO NOT EDIT
    outSizes[0] = groupTrial.innerTrial->netOutSize; // DO NOT EDIT
#endif

    InternalConnexion_G::decayParametersInitialValue = .3f;

    PopulationEvolutionParameters params;
    params.nSpecimens = 128 * nAgentsPerGroup; 
    params.nTrialsPerGroup = 2;
    params.maxNParents = 10;
    params.nLayers = nLayers;
    params.inSizes = inSizes;
    params.outSizes = outSizes;
    params.nChildrenPerLayer = nChildrenPerLayer;
    params.nEvolvedModulesPerLayer = nEvolvedModulesPerLayer;
    params.moduleReplacedFractions = moduleReplacedFractions; //in [0,.5]
    params.networkReplacedFraction = .2f; //in [0,.5]
    params.voteValue = .3f; // > 0
    params.accumulatedFitnessDecay = .9f; //in [0,1]
    params.baseMutationProbability = 1.0f;//in [0,1]
    params.consanguinityDistance = 3;  // MUST BE >= 1


    int nSteps = 10000;

    
    Population population(&groupTrial, params);

    population.evolve(nSteps);

    return 0;
}
