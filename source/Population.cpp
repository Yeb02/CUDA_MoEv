#pragma once

#include <iostream>
#include <algorithm> // std::sort
#include <fstream>   // saving (serialized)
#include <chrono>    // time since 1970 for coarse perf. measures

#include "Population.h"



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


void Population::saveFittestSpecimen()
{
	uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();

	std::ofstream os("models\\topNet_" + std::to_string(ms) + ".renon", std::ios::binary);
	networks[fittestSpecimen]->save(os);
}


Population::Population(GroupTrial* trial, PopulationEvolutionParameters& params) :
	groupTrial(trial)
{
	setEvolutionParameters(params);

	nGroups = nSpecimens / groupTrial->nAgents;

	
	fittestSpecimen = 0;


	modules.reserve(nLayers);
	modules.resize(nLayers);

	phylogeneticTrees.resize(nLayers);
	phylogeneticTrees.reserve(nLayers);


	for (int i = 0; i < nLayers; i++) 
	{
		modules[i] = new Node_G*[nEvolvedModulesPerLayer[i]];
		phylogeneticTrees[i] = new PhylogeneticNode*[nEvolvedModulesPerLayer[i]];
		for (int j = 0; j < nEvolvedModulesPerLayer[i]; j++) {

			modules[i][j] = new Node_G(inSizes+i, outSizes+i, nChildrenPerLayer[i]);
			modules[i][j]->isInModuleArray = true;

			phylogeneticTrees[i][j] = new PhylogeneticNode(nullptr, j);
		}
	}


	int probabilitiesSize = nSpecimens;
	for (int l = 0; l < nLayers; l++)
	{
		if (nEvolvedModulesPerLayer[l] > probabilitiesSize) {
			probabilitiesSize = nEvolvedModulesPerLayer[l];
		}
	}
	probabilities = std::make_unique<float[]>(probabilitiesSize);


	mutationsProbabilitiesPerLayer = std::make_unique<float[]>(nLayers);
	for (int l = 0; l < nLayers; l++)
	{
		// TODO better
		//mutationsProbabilitiesPerLayer[l] = BASE_MUTATION_P * powf((float)modules[l][0]->getNParameters(), -.5f); 
		mutationsProbabilitiesPerLayer[l] = BASE_MUTATION_P / log2f((float)modules[l][0]->getNParameters()); 
	}

	nNodesPerNetwork = 0;
	inputArraySize = 0;
	destinationArraySize = 0;
	nModulesPerNetworkLayer = std::make_unique<int[]>(nLayers);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < nLayers; l++)
	{
		int nc = nChildrenPerLayer[l];
		if (l < nLayers - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cIs = nc == 0 ? 0 : inSizes[l + 1];
		destinationArraySize += nModulesPerNetworkLayer[l] *
			(outSizes[l] + MODULATION_VECTOR_SIZE + cIs * nc);

		int cOs = nc == 0 ? 0 : outSizes[l + 1];
		inputArraySize += nModulesPerNetworkLayer[l] *
			(inSizes[l] + MODULATION_VECTOR_SIZE + cOs * nc);

		nNodesPerNetwork += nModulesPerNetworkLayer[l];
	}

	groupFitnesses = new float[nGroups * nTrialsPerGroup];

	networks = new Network*[nSpecimens];
	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = new Network(inSizes, outSizes, nChildrenPerLayer, nLayers, nTrialsPerGroup);
		createPhenotype(networks[i]);
	}

	currentModuleReplacementTreshold = std::make_unique<float[]>(nLayers);
	for (int l = 0; l < nLayers; l++) currentModuleReplacementTreshold[l] = -.7f;  // TODO arbitrary
	currentNetworkReplacementTreshold = -.7f;
}


Population::~Population() 
{
	for (int i = 0; i < nLayers; i++)
	{
		for (int j = 0; j < nEvolvedModulesPerLayer[i]; j++) {
			delete modules[i][j];
			delete phylogeneticTrees[i][j];
		}
		delete[] modules[i];

		delete[] phylogeneticTrees[i];
	}

	for (int i = 0; i < nSpecimens; i) {
		delete networks[i];
	}
	delete[] networks;

	delete[] groupFitnesses;

	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		delete toBeDestroyedModules[i];
	}
}


void Population::evolve(int nSteps) 
{
	// increasing these 2 parameters refines the fitness computations, at the cost of compute.
	// (linear in nShuffles * nNetworksSteps)

#ifdef NO_GROUP
	const int nShuffles = 1; // DO NOT CHANGE
#else 
	const int nShuffles = 5;
#endif

	const int nNetworksSteps = 5;

	for (int i = 0; i < nSteps; i++) 
	{
		
		for (int j = 0; j < nNetworksSteps; j++)
		{
			// Sort networks by age
			auto f = [](Network* n1, Network* n2) {return n1->nExperiencedTrials > n2->nExperiencedTrials; };
			std::sort(networks, networks + nSpecimens, f);


			for (int k = 0; k < nShuffles; k++)
			{
				// Shuffle networks inside each age group
				for (int i = 0; i < groupTrial->nAgents; i++) {
					std::shuffle(networks + i * nGroups, networks + (i + 1) * nGroups, generator);
				}

				bool log = (k + j) == 0;
				if (log) std::cout << "Step " << i << " ";

				evaluateGroups(log); // TODO would be interesting at each step in case of grouped trials.
			}

			replaceNetworks();
		}

		replaceModules();

		deleteUnusedModules();
	} 
}


void Population::evaluateGroups(bool log)
{
	Network** nets = new Network*[groupTrial->nAgents];

	// Pointers to the group's network's outputs.
	float** outputs = new float*[groupTrial->nAgents];

	float* netInput = new float[groupTrial->netInSize];

	for (int g = 0; g < nGroups; g++) 
	{
		for (int i = 0; i < groupTrial->nAgents; i++)
		{
			nets[i] = networks[g + nGroups * i];
			nets[i]->groupID = g;

			// }
			// we could shuffle nets at this point, so that the age of the agents is not always 
			// in the same order in a net's input / output. Probably not worth the trouble.
			// for (int i = 0; i < groupTrial->nAgents; i++) {
			
			outputs[i] = nets[i]->getOutput();
		}

		// TODO allocate the groups matrices on GPU

		groupTrial->newGroup(outputs);

		for (int i = 0; i < nTrialsPerGroup; i++) 
		{
			for (int j = 0; j < groupTrial->nAgents; j++) 
			{
				nets[j]->preTrialReset();
			}

			groupTrial->intraGroupReset();
			
			while (!groupTrial->innerTrial->isTrialOver) 
			{
#ifdef NO_GROUP
				nets[0]->step(groupTrial->innerTrial->observations.data());
				groupTrial->innerTrial->step(outputs[0]);
#else
				std::copy(
					groupTrial->innerTrial->observations.begin(),
					groupTrial->innerTrial->observations.end(),
					netInput
				);
				for (int j = 0; j < groupTrial->nAgents; j++)
				{
					groupTrial->prepareInput(netInput + groupTrial->innerTrial->netInSize, j);
					nets[j]->step(netInput);
				}
				groupTrial->step();
#endif
			}

			groupTrial->normalizeVotes();
			for (int j = 0; j < groupTrial->nAgents; j++)
			{
				nets[j]->perTrialVotes[i] = groupTrial->accumulatedVotes[j];
			}

			groupFitnesses[i * nGroups + g] = groupTrial->innerTrial->score;

		}

		// TODO deallocate the groups matrices on GPU
	}

	// monitoring
	if (log) {
		float avgF = 0.0f;
		float maxF = -100000.0f;
		float* avgGroupF = new float[nGroups];
		std::fill(avgGroupF, avgGroupF + nGroups, 0.0f);
		for (int i = 0; i < nTrialsPerGroup * nGroups; i++)
		{
			if (groupFitnesses[i] > maxF) maxF = groupFitnesses[i];
			avgF += groupFitnesses[i];
			avgGroupF[i % nGroups] += groupFitnesses[i];
		}
		avgF /= (float)(nTrialsPerGroup * nGroups);
		float maxAvgGroupF = -100000.0f;
		for (int i = 0; i < nGroups; i++)
		{
			if (maxAvgGroupF < avgGroupF[i]) maxAvgGroupF = avgGroupF[i];
		}
		maxAvgGroupF /= nTrialsPerGroup;
		delete[] avgGroupF;

		// scores are per group.
		std::cout << " Over groups, avg avg score: " << avgF << ", best avg score " << maxAvgGroupF << ", best score : " << maxF << std::endl;
	}



	// For all i, the i-th trials of every groups are compared (ranked).
	for (int i = 0; i < nTrialsPerGroup; i++)
	{
		rankArray(groupFitnesses + i * nGroups, groupFitnesses + i * nGroups, nGroups);
	}

	// Kinda ugly, possibly 100% cache misses. TODO make lifetimeFitness,
	// tempFitnessAccumulator and nTempFitnessAccumulations into
	// population owned arrays, adding a field arrayId to Node_G.
	// And accumulate only if isInModuleArray. (cache miss....)
	for (int l = 0; l < nLayers; l++) {
		for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
			modules[l][i]->tempFitnessAccumulator = 0.0f;
			modules[l][i]->nTempFitnessAccumulations = 0;
		}
	}

	// update modules and network fitnesses.
	for (int i = 0; i < nSpecimens; i++)
	{
		float f = 0.0f;
		for (int j = 0; j < nTrialsPerGroup; j++)
		{
			float gf = groupFitnesses[j * nGroups + networks[i]->groupID];
			float v = networks[i]->perTrialVotes[j];

#ifdef NO_GROUP
			f += gf;
#else
			f += (1.0f + voteValue * (gf>0?1.f:-1.f) * v) * gf;
#endif
		}
		f /= (float)nTrialsPerGroup;

		networks[i]->lifetimeFitness = networks[i]->lifetimeFitness * accumulatedFitnessDecay + f;		

		for (int j = 0; j < nNodesPerNetwork; j++) {
			networks[i]->nodes[j]->tempFitnessAccumulator += f;
			networks[i]->nodes[j]->nTempFitnessAccumulations++;
		}
 	}

	for (int l = 0; l < nLayers; l++) {
		for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
			Node_G* m = modules[l][i];
			if (m->nTempFitnessAccumulations == 0) continue;
			m->tempFitnessAccumulator /= (float) m->nTempFitnessAccumulations;
			m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay + m->tempFitnessAccumulator;		
		}
	}

	delete[] nets;
	delete[] outputs;
	delete[] netInput;
}


void Population::createPhenotype(Network* net) 
{
	Node_G** genotype = new Node_G * [nNodesPerNetwork]; // ownership is transferred to the network.


	int id = 0;
	for (int l = 0; l < nLayers; l++) {
		for (int n = 0; n < nModulesPerNetworkLayer[l]; n++) {
			genotype[id] = modules[l][INT_0X(nEvolvedModulesPerLayer[l])];
			genotype[id]->nUsesInNetworks++;
			id++;
		}
	}

	net->createPhenotype(inputArraySize, destinationArraySize, genotype);
}


Node_G* Population::createChild(PhylogeneticNode* primaryParent, int moduleLayer) {

	// Reminder: when this function is called, nParents > 1.

	std::vector<int> parents;
	parents.push_back(primaryParent->modulePosition);
	
	std::vector<float> rawWeights;
	rawWeights.resize(maxNParents);

	// fill parents, and weights are initialized with a function of the phylogenetic distance.
	{
		bool listFilled = false;

		auto distanceValue = [] (float d) 
		{
			return powf(1.0f+d, -0.6f);
		};

		rawWeights[0] = 1.0f; // = distanceValue(0)

		for (int depth = CONSANGUINITY_DISTANCE; depth < MAX_PHYLOGENETIC_DEPTH; depth++) {

			float w = distanceValue((float) depth);
			std::fill(rawWeights.begin() + (int)parents.size(), rawWeights.end(), w);

			primaryParent->fillList(parents, 0, depth, nullptr, maxNParents);

			// parents.size() == nParents can happen if there were exactly as many parents 
			// as there was room left in the array. This does not set listFilled = true.
			if (listFilled || parents.size() == maxNParents) break;
		}
	}

	if (parents.size() == 1) { // There were no close enough relatives.
		return new Node_G(modules[moduleLayer][primaryParent->modulePosition]);
	}

	float f0 = modules[moduleLayer][primaryParent->modulePosition]->lifetimeFitness;

	std::vector<Node_G*> parentNodes;

#ifdef  SPARSE_MUTATION_AND_COMBINATIONS
	parentNodes.push_back(modules[moduleLayer][parents[0]]);
	for (int i = 1; i < parents.size(); i++) {
		if (modules[moduleLayer][parents[i]]->lifetimeFitness > f0) {
			parentNodes.push_back(modules[moduleLayer][parents[i]]);
		}
	}
	if (parentNodes.size() == 1) { // There were no close enough relatives with a better fitness
		return new Node_G(modules[moduleLayer][primaryParent->modulePosition]);
	}
#else

	parentNetworks.resize(parents.size());
	for (int i = 0; i < parents.size(); i++) {
		parentNetworks[i] = networks[parents[i]];
	}

	// weights are an increasing function of the relative fitness.
	{
		rawWeights[0] = 1.0f;
		for (int i = 1; i < parents.size(); i++) {
			rawWeights[i] *= (fitnesses[parents[i]] - f0); // TODO  adapt to MoEv
		}
	}
#endif
	

	return Node_G::combine(parentNodes.data(), rawWeights.data(), (int)parentNodes.size());
}


void Population::deleteUnusedModules()
{
	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		if (toBeDestroyedModules[i]->nUsesInNetworks == 0) {
			delete toBeDestroyedModules[i];

			// there may be cases where delete does not already do it :
			toBeDestroyedModules[i] = nullptr; 
		}
	}
	auto eraseUnused = [](Node_G* n) {
		return n == nullptr;
	};

	std::erase_if(toBeDestroyedModules, eraseUnused);
}


void Population::replaceNetworks()
{
	int nReplacements = 0;
	for (int i = 0; i < nSpecimens; i++) {
		if (networks[i]->lifetimeFitness > currentNetworkReplacementTreshold) continue;

		nReplacements++;
		int id = 0;
		for (int l = 0; l < nLayers; l++) {
			for (int n = 0; n < nModulesPerNetworkLayer[l]; n++) {
				networks[i]->nodes[id]->nUsesInNetworks--;
				id++;
			}
		}

		delete networks[i];

		networks[i] = new Network(inSizes, outSizes, nChildrenPerLayer, nLayers, nTrialsPerGroup);
		createPhenotype(networks[i]);
	}

	if ((float)nReplacements / (float)nSpecimens > networkReplacedFraction) 
	{
		currentNetworkReplacementTreshold *= 1.0f / .8f; // TODO arbitrary
	}
	else {
		currentNetworkReplacementTreshold *= .8f;
	}
}


void Population::replaceModules() {
	uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	for (int l = 0; l < nLayers; l++) {

		// Compute roulette probabilities.
		{
			float invProbaSum = 0.0f;

			for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
				float pRaw = modules[l][i]->lifetimeFitness - 0.0f;
				if (pRaw > 0) probabilities[i] = pRaw;
				else probabilities[i] = 0.0f;

				invProbaSum += probabilities[i];
			}

			invProbaSum = 1.0f / invProbaSum;

			probabilities[0] = probabilities[0] * invProbaSum;
			for (int i = 1; i < nEvolvedModulesPerLayer[l]; i++) {
				probabilities[i] = probabilities[i - 1] + probabilities[i] * invProbaSum;
			}
		}

		int nReplacements = 0;

		for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
			if (modules[l][i]->lifetimeFitness < currentModuleReplacementTreshold[l]) 
			{
				modules[l][i]->isInModuleArray = false;
				toBeDestroyedModules.push_back(modules[l][i]);

				phylogeneticTrees[l][i]->onModuleDestroyed();

				int parentID = binarySearch(probabilities.get(), UNIFORM_01, nEvolvedModulesPerLayer[l]);

				modules[l][i] = createChild(phylogeneticTrees[l][parentID], l);
				modules[l][i]->mutateFloats(mutationsProbabilitiesPerLayer[l]);
				modules[l][i]->isInModuleArray = true;

				phylogeneticTrees[l][i] = new PhylogeneticNode(phylogeneticTrees[l][parentID], i);

				nReplacements++;
			}
		}
		//std::cout << (float)nReplacements / (float)nEvolvedModulesPerLayer[l] << " ";
		if ((float)nReplacements / (float)nEvolvedModulesPerLayer[l] > moduleReplacedFractions[l])
		{
			currentModuleReplacementTreshold[l] *= 1.0f / .8f; 
		}
		else {
			currentModuleReplacementTreshold[l] *= .8f;
		}
	}
	//std::cout << std::endl;

	uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//std::cout << "Modules creation took " << stop - start << " ms." << std::endl;
}
