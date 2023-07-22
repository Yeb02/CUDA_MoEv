#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

#endif

#include <iostream>
#include "Population.h"
#include "Random.h"

#ifdef ROCKET_SIM_T
#include "RocketSim.h"
#endif


#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;



int main()
{
    LOG("Seed : " << seed);

#ifdef ROCKET_SIM_T
    // Path to where you dumped rocket league collision meshes.
    RocketSim::Init((std::filesystem::path)"C:/Users/alpha/Bureau/RLRL/collisionDumper/x64/Release/collision_meshes");
#endif

    int channelSize = 2;
    int nAgentsPerTrial = 4;

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

    GroupTrial groupTrial(innerTrial, nAgentsPerTrial, channelSize);

 
    const int nLayers = 3;
    int inSizes[nLayers] = { groupTrial.netInSize, 8, 4};
    int outSizes[nLayers] = { groupTrial.netOutSize, 7, 3};
    int nChildrenPerLayer[nLayers] = {2, 1, 0};
    int nEvolvedModulesPerLayer[nLayers] = {32, 32, 32};
    float moduleReplacedFractions[nLayers] = {.3f, .3f, .3f};

    PopulationEvolutionParameters params;
    params.nSpecimens = 16 * nAgentsPerTrial; //16 -> 512 in most cases;
    params.nTrialsPerGroup = 3;
    params.nParents = 10;
    params.nLayers = nLayers;
    params.inSizes = inSizes;
    params.outSizes = outSizes;
    params.nChildrenPerLayer = nChildrenPerLayer;
    params.nEvolvedModulesPerLayer = nEvolvedModulesPerLayer;
    params.moduleReplacedFractions = moduleReplacedFractions;
    params.networkReplacedFraction = .2f;
    params.voteValue = .3f;
    params.accumulatedFitnessDecay = .9f;


    int nSteps = 10000;

    
    Population population(&groupTrial, params);

    population.evolve(nSteps);

    // Tests.
    /*if (false) {
        std::ifstream is("models\\topNet_1685971637922_631.renon", std::ios::binary);
        LOG(is.is_open());
        Network* n = new Network(is);
        LOG("Loaded.");
        n->createPhenotype();
        n->preTrialReset();
        trials[0]->reset(true);
        float avg_thr = 0.0f;
        while (!trials[0]->isTrialOver) {
            n->step(trials[0]->observations);
            trials[0]->step(n->getOutput());
            LOG(n->getOutput()[0]);
        }
        delete n;
        LOG("Reloaded best specimen's score on the same trial = " << trials[0]->score);
    }*/

    return 0;
}
