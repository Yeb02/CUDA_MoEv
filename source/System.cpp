#pragma once

#include <iostream>
#include <algorithm> // std::sort
#include <fstream>   // saving (serialized)
#include <chrono>    // time since 1970 for coarse perf. measures

#include <mutex>			  // threading
#include <condition_variable> // threading

// Threading utils
std::mutex m;
std::condition_variable startProcessing;
std::condition_variable doneProcessing;
std::condition_variable hasTerminated;
bool allStartProcessing = false;
int nTerminated = 0;
int nDoneProcessing = 0;


#include "System.h"



// src is unchanged.
void normalizeArray(float* src, float* dst, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += src[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		dst[i] = src[i] - avg;
		variance += dst[i] * dst[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float) size);
	for (int i = 0; i < size; i++) {
		dst[i] *= InvStddev;
	}
}

// src is unchanged. Results in [-1, 1]
void rankArray(float* src, float* dst, int size) {
	std::vector<int> positions(size);
	for (int i = 0; i < size; i++) {
		positions[i] = i;
	}
	// sort position by ascending value.
	std::sort(positions.begin(), positions.end(), [src](int a, int b) -> bool
		{
			return src[a] < src[b];
		}
	);
	float invSize = 1.0f / (float)size;
	for (int i = 0; i < size; i++) {
		// linear in [-1,1], -1 for the worst specimen, 1 for the best
		float positionValue = (float)(2 * i - size) * invSize;
		// arbitrary, to make it a bit more selective. 
		positionValue = 1.953f * powf(positionValue * .8f, 3.0f);

		dst[positions[i]] = positionValue;
	}
	return;
}



System::System(Trial** _trials, SystemEvolutionParameters& sParams, NetworkParameters& nParams, ModulePopulationParameters& mpParams, int _nThreads) :
	trials(_trials), nThreads(_nThreads)
{
	setEvolutionParameters(sParams);

	startThreads();

	//noveltyEncoder = std::make_unique<NoveltyEncoder>();
	

	fittestSpecimen = 0;


	// TODO -.7 arbitrary. As it only affects initialization, it can stay here, but
	// work of satisfactory values must be done.
	currentAgentReplacementTreshold = -.7f; // <0




	nModulesPerAgent = 0;

	int inputArraySize = 0;
	int destinationArraySize = 0;
	nModulesPerNetworkLayer.resize(nParams.nLayers);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < nParams.nLayers; l++)
	{
		int nc = nParams.nChildrenPerLayer[l];
		if (l < nParams.nLayers - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cIs = nc == 0 ? 0 : nParams.inSizes[l + 1];
		destinationArraySize += nModulesPerNetworkLayer[l] *
			(nParams.outSizes[l] + cIs * nc);

		int cOs = nc == 0 ? 0 : nParams.outSizes[l + 1];
		inputArraySize += nModulesPerNetworkLayer[l] *
			(nParams.inSizes[l] + cOs * nc);

		nModulesPerAgent += nModulesPerNetworkLayer[l];

		mpParams.nModules = sParams.nEvolvedModulesPerLayer[l];

		Node_GFixedParameters nfp(&(nParams.inSizes[l]), &(nParams.outSizes[l]), &(nParams.nChildrenPerLayer[l]));
		populations[l] = new ModulePopulation<Node_G>(mpParams, static_cast<IModuleFixedParameters*>(&nfp));

	}
	


	Network::destinationArraySize = destinationArraySize;
	Network::inputArraySize = inputArraySize;
	Network::inS = nParams.inSizes;
	Network::outS = nParams.outSizes;
	Network::nC = nParams.nChildrenPerLayer;
	Network::nLayers = nParams.nLayers;

	agents = new IAgent*[nAgents];
	for (int i = 0; i < nAgents; i++) {
		agents[i] = static_cast<IAgent*>(new Network(nModulesPerAgent));
		static_cast<Network*>(agents[i])->createPhenotype(populations); // This is so stupid TODO aaaaaaaaa
	}
}


System::~System() 
{
	for (int i = 0; i < populations.size(); i) {
		delete populations[i];
	}
	for (int i = 0; i < nAgents; i) {
		delete agents[i];
	}
	delete[] agents;

}


void System::saveFittestSpecimen()
{
	uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	std::ofstream os("models\\topNet_" + std::to_string(ms) + ".renon", std::ios::binary);
	agents[fittestSpecimen]->save(os);
}



void System::startThreads()
{
	stopThreads();

	mustTerminate = false;
	
	threads.resize(0); // to destroy all previously existing threads.

	if (nThreads < 1) return;

	threads.reserve(nThreads);
	threadIteration = -1;
	int nAgentsPerThread = nAgents / nThreads;
	for (int t = 0; t < nThreads; t++) {
		threads.emplace_back(&System::perThreadMainLoop, this, t);
	}
}


void System::stopThreads()
{
	if (threads.size() < 1) return;

	{
		std::lock_guard<std::mutex> lg(m);
		mustTerminate = true;
		nTerminated = (int)threads.size();
	}
	startProcessing.notify_all();

	// wait on workers.
	{
		std::unique_lock<std::mutex> lg(m);
		hasTerminated.wait(lg, [] {return nTerminated == 0; });
	}

	for (int t = 0; t < nThreads; t++) threads[t].join();
}


void System::perThreadMainLoop(const int threadID)
{
	int currentThreadIteration = 0;

	Trial* trial = trials[threadID];

	while (true) {
		std::unique_lock<std::mutex> ul(m);
		startProcessing.wait(ul, [&currentThreadIteration, this] {return (currentThreadIteration == threadIteration) || mustTerminate; });
		if (mustTerminate) {
			nTerminated--;
			if (nTerminated == 0) {
				ul.unlock();
				hasTerminated.notify_one();
			}
			break;
		}
		ul.unlock();


		currentThreadIteration++;

		for (int a = threadID * nAgentsPerThread; a < (threadID + 1) * nAgentsPerThread; a++)
		{
			for (int i = 0; i < nTrialsPerNetworkCycle; i++)
			{
				trial->reset(false);
				agents[a]->preTrialReset();

				while (!trial->isTrialOver)
				{
					agents[a]->step(trial->observations.data());
					trial->step(agents[a]->getOutput());
				}

				fitness = trial->score;
			}
		}

		ul.lock();
		nDoneProcessing--;
		if (nDoneProcessing == 0) {
			ul.unlock();
			doneProcessing.notify_one();
		}
	}
}


void System::evolve(int nSteps) 
{
	for (int i = 0; i < nSteps; i++) 
	{
		
		for (int j = 0; j < nNetworksCyclesPerModuleCycle; j++)
		{
			// send msg to threads to evaluate their agents
			{
				std::lock_guard<std::mutex> lg(m);
				nDoneProcessing = nThreads;
				threadIteration++;
			}
			startProcessing.notify_all();

			// wait on threads.
			{
				std::unique_lock<std::mutex> lg(m);
				doneProcessing.wait(lg, [] {return nDoneProcessing == 0; });
			}

			replaceNetworks();
		}


		uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		for (int p = 0; p < populations.size(); p++) {
			populations[p]->replaceModules();
		}

		uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		std::cout << "Step " << i;
		std::cout << ",  Modules replacement took " << stop - start << " ms." << std::endl;
	} 
}



void System::accumulateModuleFitnesses()
{

	for (int i = 0; i < nTrialsPerGroup; i++)
	{
		rankArray(groupFitnesses + i * nSpecimens, groupFitnesses + i * nSpecimens, nSpecimens);
	}

	// update modules and network fitnesses.
	for (int i = 0; i < nSpecimens; i++)
	{
		float f = 0.0f;
		for (int j = 0; j < nTrialsPerGroup; j++)
		{
			f += groupFitnesses[j * nSpecimens + i];
		}
		f /= (float)nTrialsPerGroup;

		networks[i]->lifetimeFitness = networks[i]->lifetimeFitness * accumulatedFitnessDecay + f;

		for (int j = 0; j < nNodesPerNetwork; j++) {
			networks[i]->nodes[j]->tempFitnessAccumulator += f;
			networks[i]->nodes[j]->nTempFitnessAccumulations++;
		}
		if (useInMLP) {
			networks[i]->inMLP->type->tempFitnessAccumulator += f;
			networks[i]->inMLP->type->nTempFitnessAccumulations++;
		}
		if (useOutMLP) {
			networks[i]->outMLP->type->tempFitnessAccumulator += f;
			networks[i]->outMLP->type->nTempFitnessAccumulations++;
		}
	}
}



void System::replaceNetworks()
{
	int nReplacements = 0;
	for (int i = 0; i < nAgents; i++) {

		if (agents[i]->lifetimeFitness > currentAgentReplacementTreshold) continue;


		// Replace this network

		nReplacements++;

		delete agents[i];


		agents[i] = static_cast<IAgent*>(new Network(nModulesPerAgent));
		static_cast<Network*>(agents[i])->createPhenotype(populations); // This is so stupid TODO aaaaaaaaa
	}


	// TODO .8 arbitrary
	if ((float)nReplacements / (float)nAgents > agentsReplacedFraction)  
	{
		currentAgentReplacementTreshold = std::max(-10.0f, currentAgentReplacementTreshold / .8f);
	}
	else {
		currentAgentReplacementTreshold = std::min(-.03f, currentAgentReplacementTreshold * .8f);
	}
}

