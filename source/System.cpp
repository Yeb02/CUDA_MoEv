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


System::System(Trial** _trials, SystemEvolutionParameters& sParams, AGENT_PARAMETERS& aParams, ModulePopulationParameters& mpParams, int _nThreads) :
	trials(_trials), nThreads(_nThreads)
{
	setEvolutionParameters(sParams);

	startThreads();

	

	fittestSpecimen = 0;

	agentFitnesses.resize(nAgents);
	teachers.resize(nSupervisedTrialsPerAgentCycle);

	
	currentAgentReplacementTreshold = -(1.0f - accumulatedFitnessDecay) * agentsReplacedFraction * 2.0f; // <0

	agentsScores.resize(nEvaluationTrialsPerAgentCycle);
	for (int i = 0; i < nEvaluationTrialsPerAgentCycle; i++) {
		agentsScores[i] = new float[nAgents];
	}


	nModulesPerAgent = 0;

	populations.resize(aParams.nLayers);

	int inCoutSize = 0;
	int outCinSize = 0;
	int activationArraySize = aParams.outSizes[0];
	std::vector<int> nModulesPerNetworkLayer(aParams.nLayers);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < aParams.nLayers; l++)
	{
		int nc = aParams.nChildrenPerLayer[l];
		if (l < aParams.nLayers - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cIs = nc == 0 ? 0 : aParams.inSizes[l + 1];
		outCinSize += nModulesPerNetworkLayer[l] *
			(aParams.outSizes[l] + cIs * nc);

		int cOs = nc == 0 ? 0 : aParams.outSizes[l + 1];
		inCoutSize += nModulesPerNetworkLayer[l] *
			(aParams.inSizes[l] + cOs * nc);

		activationArraySize += nModulesPerNetworkLayer[l] * (aParams.inSizes[l] + cOs * nc);

		nModulesPerAgent += nModulesPerNetworkLayer[l];

		mpParams.nModules = sParams.nEvolvedModulesPerLayer[l];


		MODULE_PARAMETERS nfp(&(aParams.inSizes[l]), &(aParams.outSizes[l]), &(aParams.nChildrenPerLayer[l]));
		populations[l] = new ModulePopulation(mpParams, nfp);


	}
	

#ifdef PREDICTIVE_CODING
	AGENT::activationArraySize = activationArraySize;
#else
	AGENT::outCinSize = outCinSize;
	AGENT::inCoutSize = inCoutSize;
#endif
	AGENT::inS = aParams.inSizes;
	AGENT::outS = aParams.outSizes;
	AGENT::nC = aParams.nChildrenPerLayer;
	AGENT::nLayers = aParams.nLayers;

	agents = new AGENT*[nAgents];
	for (int i = 0; i < nAgents; i++) {
		agents[i] = new AGENT(nModulesPerAgent);
		agents[i]->createPhenotype(populations); 
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


	for (int i = 0; i < nEvaluationTrialsPerAgentCycle; i++) {
		delete agentsScores[i];
	}
	
	for (int i = 0; i < nSupervisedTrialsPerAgentCycle; i++) {
		delete teachers[i];
	}


}


void System::saveBestAgent()
{
	uint64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	std::ofstream os("models\\bestAgent_" + std::to_string(now_ms) + ".moeva", std::ios::binary);
	agents[fittestSpecimen]->save(os);
}


void System::log(int step) 
{
	float avg_avg_s = 0.0f;
	float max_avg_s = -10000000.0f;
	float max_s = -1000000.0f;
	for (int j = 0; j < nAgents ; j++)
	{
		float ss = 0.0f;
		for (int k = 0; k < nAgentCyclesPerModuleCycle; k++)
		{
			float s = agentsScores[k][j];
			ss += s;
			max_s = s > max_s ? s : max_s;
		}
		ss /= (float)nAgentCyclesPerModuleCycle;
		max_avg_s = ss > max_avg_s ? ss : max_avg_s;
		avg_avg_s += ss;
	}
	avg_avg_s /= (float)nAgents;
	std::cout << "At step " << step << ", max score : " << max_s << ", best agent avg score : " 
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



		// teaching/ supervised learning phase
		for (int t = 0; t < nSupervisedTrialsPerAgentCycle; t++) 
		{

			AGENT* teacher = teachers[(threadID + t) % nSupervisedTrialsPerAgentCycle];
			
			float* teacherOutput = teacher->getOutput();
			
			teacher->preTrialReset();
			for (int a = threadID * nAgentsPerThread; a < (threadID + 1) * nAgentsPerThread; a++) {
				agents[a]->preTrialReset();
			}

			trial->reset(false);

			while (!trial->isTrialOver)
			{
				teacher->step(trial->observations.data(), false);
				for (int a = threadID * nAgentsPerThread; a < (threadID + 1) * nAgentsPerThread; a++)
				{
					agents[a]->step(trial->observations.data(), true, teacherOutput);
				}

				trial->step(teacherOutput);
			}

			ul.lock();
			nDoneProcessing--;
			if (nDoneProcessing == 0) {
				ul.unlock();
				doneProcessing.notify_one();
				ul.lock();
			}

			
			startProcessing.wait(ul, [&currentThreadIteration, this] {return (currentThreadIteration == threadIteration); });
			ul.unlock();
			currentThreadIteration++;
		}



		
		// evaluation phase
		for (int a = threadID * nAgentsPerThread; a < (threadID + 1) * nAgentsPerThread; a++)
		{
			for (int i = 0; i < nEvaluationTrialsPerAgentCycle; i++)
			{
				trial->reset(false);
				agents[a]->preTrialReset();

				while (!trial->isTrialOver)
				{
					agents[a]->step(trial->observations.data(), false);
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
	// small hack because their are no teachers at the very first agent cycle.
	int tmp_nSupervisedTrialsPerAgentCycle = nSupervisedTrialsPerAgentCycle;
	nSupervisedTrialsPerAgentCycle = 0;

	for (int s = 0; s < nSteps; s++) 
	{
		

		for (int j = 0; j < nAgentCyclesPerModuleCycle; j++)
		{
			// zero agent score accumulators
			for (int i = 0; i < nEvaluationTrialsPerAgentCycle; i++)
			{
				std::fill(agentsScores[i], agentsScores[i] + nAgents, .0f);
			}

			// + 1 for the evaluation part.
			for (int i = 0; i < nSupervisedTrialsPerAgentCycle + 1; i++)
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
			}

			// see hack at the beginning of this function
			if ((s + j) == 0)
			{
				nSupervisedTrialsPerAgentCycle = tmp_nSupervisedTrialsPerAgentCycle;
			}

			if (j == nAgentCyclesPerModuleCycle-1) log(s);

			replaceAgents();
		}

		uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		for (int p = 0; p < populations.size(); p++) {
			populations[p]->replaceModules();
		}

		uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		//std::cout << "Step " << s;
		//std::cout << ",  Modules replacement took " << stop - start << " ms." << std::endl;
	} 
}


void System::replaceAgents()
{

	for (int i = 0; i < nEvaluationTrialsPerAgentCycle; i++) 
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

		// the exponential avg starts at 0 for "newborns"
		float fa = agents[i]->lifetimeFitness;

		float fm = agentsScores[0][i];
		for (int j = 0; j < nEvaluationTrialsPerAgentCycle; j++)
		{
			float ds = (1.0F - accumulatedFitnessDecay) * agentsScores[j][i];
			fa = fa * accumulatedFitnessDecay + ds;
			fm = fm * accumulatedFitnessDecay + ds;
		}

		agents[i]->accumulateFitnessInModules(fm);

		agents[i]->lifetimeFitness = fa;

		// if its fitness is too low, the agent is to be replaced  
		if (agents[i]->lifetimeFitness <= currentAgentReplacementTreshold) 
		{
			nReplacements++;

			delete agents[i];

			agents[i] = new AGENT(nModulesPerAgent);
			agents[i]->createPhenotype(populations);
		}
		
		agentFitnesses[i] = agents[i]->lifetimeFitness;
	}


	// TODO .8f is arbitrary
	if ((float)nReplacements / (float)nAgents > agentsReplacedFraction)  
	{
		currentAgentReplacementTreshold = currentAgentReplacementTreshold / .8f;
	}
	else {
		currentAgentReplacementTreshold = currentAgentReplacementTreshold * .8f;
	}


	// Determine teachers.
	{
		std::vector<int> positions(nAgents);
		for (int i = 0; i < nAgents; i++) {
			positions[i] = i;
		}
		float* src = agentFitnesses.data();

		// sort "positions" by ascending fitnesses.
		std::sort(positions.begin(), positions.end(), [src](int a, int b) -> bool
			{
				return src[a] < src[b];
			}
		);

		for (int i = 0; i < nSupervisedTrialsPerAgentCycle; i++) {
			delete teachers[i];
			teachers[i] = new AGENT(*(agents[positions[nAgents - 1 - i]]));
		}
	}
}

