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


// src is unchanged. src can be the same array as dst.
// Dst has mean 0 and variance 1.
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

// src is unchanged. src can be the same array as dst.
// Dst values in [-1, 1], -1 attibuted to the worst of src and 1 to the best.
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


	// TODO -.7 arbitrary. As it only affects initialization, it can stay here
	currentAgentReplacementTreshold = -.7f; // <0

	agentsScores.resize(nTrialsPerNetworkCycle);
	for (int i = 0; i < nTrialsPerNetworkCycle; i++) {
		agentsScores[i] = new float[nAgents];
	}


	nModulesPerAgent = 0;

	populations.resize(nParams.nLayers);

	int inputArraySize = 0;
	int destinationArraySize = 0;
	std::vector<int> nModulesPerNetworkLayer(nParams.nLayers);
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
	stopThreads();

	for (int i = 0; i < populations.size(); i) {
		delete populations[i];
	}

	for (int i = 0; i < nAgents; i) {
		delete agents[i];
	}
	delete[] agents;


	for (int i = 0; i < nTrialsPerNetworkCycle; i++) {
		delete agentsScores[i];
	}


}


void System::saveBestAgent()
{
	uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	std::ofstream os("models\\bestAgent_" + std::to_string(now_ms) + ".moeva", std::ios::binary);
	agents[fittestSpecimen]->save(os);
}


void System::log() 
{
	float avg_avg_s = 0.0f;
	float max_avg_s = -10000000.0f;
	float max_s = -1000000.0f;
	for (int j = 0; j < nAgents ; j++)
	{
		float ss = 0.0f;
		for (int k = 0; k < nNetworksCyclesPerModuleCycle; k++)
		{
			float s = agentsScores[k][j];
			ss += s;
			max_s = s > max_s ? s : max_s;
		}
		ss /= (float)nNetworksCyclesPerModuleCycle;
		max_avg_s = ss > max_avg_s ? ss : max_avg_s;
		avg_avg_s += ss;
	}
	avg_avg_s /= (float)nAgents;
	std::cout << "Max score: " << max_s << ", best agent avg score: " 
		<< max_avg_s << ", avg of avg scores : " << avg_avg_s << std::endl;
}


void System::startThreads()
{
	stopThreads();

	mustTerminate = false;
	
	threads.resize(0); // to destroy all previously existing threads.

	if (nThreads < 1) return;

	threads.reserve(nThreads);
	threadIteration = -1;
	nAgentsPerThread = nAgents / nThreads;
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

				agentsScores[i][a] = trial->score;
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
			// zero agent score accumulators
			for (int i = 0; i < nTrialsPerNetworkCycle; i++)
			{
				std::fill(agentsScores[i], agentsScores[i] + nAgents, .0f);
			}

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

			if (j==0) log();

			replaceNetworks();
		}

		uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		for (int p = 0; p < populations.size(); p++) {
			populations[p]->replaceModules();
		}

		uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//std::cout << "Step " << i;
		//std::cout << ",  Modules replacement took " << stop - start << " ms." << std::endl;
	} 
}


void System::replaceNetworks()
{

	for (int i = 0; i < nTrialsPerNetworkCycle; i++) 
	{
		switch (scoreTransformation) {

		case SCORE_BATCH_TRANSFORMATION::NONE:
			break;
		case SCORE_BATCH_TRANSFORMATION::NORMALIZE:
			normalizeArray(agentsScores[i], agentsScores[i], nAgents);
			break;
		case SCORE_BATCH_TRANSFORMATION::RANK:
			rankArray(agentsScores[i], agentsScores[i], nAgents);
			break;
		} 
	}



	int nReplacements = 0;
	for (int i = 0; i < nAgents; i++) {

		float f = 0.0f;
		for (int j = 0; j < nTrialsPerNetworkCycle; j++)
		{
			float gamma = 1.0f - powf(1.5f, -(float)(j + agents[i]->nExperiencedTrials - nTrialsPerNetworkCycle));
			f += gamma * agentsScores[j][i];
		}
		f /= (float)nTrialsPerNetworkCycle;

		agents[i]->accumulateFitnessInModules(f);

		agents[i]->lifetimeFitness = agents[i]->lifetimeFitness * accumulatedFitnessDecay + f;


		// to be replaced ? if no, continue to next loop iteration.
		if (agents[i]->lifetimeFitness > currentAgentReplacementTreshold) continue;


		nReplacements++;

		delete agents[i];

		agents[i] = static_cast<IAgent*>(new Network(nModulesPerAgent));
		static_cast<Network*>(agents[i])->createPhenotype(populations); // This is so stupid TODO aaaaaaaaa
	}


	// TODO .8f is arbitrary
	if ((float)nReplacements / (float)nAgents > agentsReplacedFraction)  
	{
		currentAgentReplacementTreshold = std::max(-10.0f, currentAgentReplacementTreshold / .8f);
	}
	else {
		currentAgentReplacementTreshold = std::min(-.03f, currentAgentReplacementTreshold * .8f);
	}
}

