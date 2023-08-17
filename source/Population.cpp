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

	phylogeneticTrees.reserve(nLayers);
	phylogeneticTrees.resize(nLayers);

	for (int i = 0; i < nLayers; i++) 
	{
		modules[i] = new Node_G*[nEvolvedModulesPerLayer[i]];
		phylogeneticTrees[i] = new PhylogeneticNode*[nEvolvedModulesPerLayer[i]];
		for (int j = 0; j < nEvolvedModulesPerLayer[i]; j++) {

			modules[i][j] = new Node_G(inSizes+i, outSizes+i, nChildrenPerLayer[i]);
			modules[i][j]->isStillEvolved = true;

			phylogeneticTrees[i][j] = new PhylogeneticNode(nullptr, j);
		}
	}

	if (useInMLP) 
	{
		inMLPphylogeneticTree = new PhylogeneticNode*[nInMLPs];
		inMLPs = new MLP_G*[nInMLPs];

		for (int j = 0; j < nInMLPs; j++) {

			inMLPs[j] = new MLP_G(inMLPsizes, inMLPnLayers);
			inMLPs[j]->isStillEvolved = true;

			inMLPphylogeneticTree[j] = new PhylogeneticNode(nullptr, j);
		}
	}

	if (useOutMLP)
	{
		outMLPphylogeneticTree = new PhylogeneticNode * [nOutMLPs];
		outMLPs = new MLP_G * [nOutMLPs];

		for (int j = 0; j < nOutMLPs; j++) {

			outMLPs[j] = new MLP_G(outMLPsizes, outMLPnLayers);
			outMLPs[j]->isStillEvolved = true;

			outMLPphylogeneticTree[j] = new PhylogeneticNode(nullptr, j);
		}
	}

	groupFitnesses = new float[nGroups * nTrialsPerGroup];

	// TODO -.7 arbitrary. As it only affects initialization, it can stay here, but
	// work of satisfactory values must be done.
	currentModuleReplacementTreshold = std::make_unique<float[]>(nLayers);
	for (int l = 0; l < nLayers; l++) currentModuleReplacementTreshold[l] = -.7f;  // <0
	currentNetworkReplacementTreshold = -.7f; // <0
	currentInMLPReplacementTreshold = -.7f;// <0
	currentOutMLPReplacementTreshold = -.7f;// <0

	int probabilitiesSize = nSpecimens;
	for (int l = 0; l < nLayers; l++)
	{
		if (nEvolvedModulesPerLayer[l] > probabilitiesSize) {
			probabilitiesSize = nEvolvedModulesPerLayer[l];
		}
	}
	if (useInMLP  && probabilitiesSize < nInMLPs) probabilitiesSize = nInMLPs;
	if (useOutMLP && probabilitiesSize < nOutMLPs) probabilitiesSize = nOutMLPs;
	probabilities = std::make_unique<float[]>(probabilitiesSize);


	mutationsProbabilitiesPerLayer = std::make_unique<float[]>(nLayers);
	for (int l = 0; l < nLayers; l++)
	{
		// TODO better. The goal here is to lower the rate of mutation as the number of parameters grow.
		
		mutationsProbabilitiesPerLayer[l] = baseMutationProbability 
			/ log2f((float)modules[l][0]->getNParameters());
			// / (log2f(powf(2.0f, 8.0f) + (float)modules[l][0]->getNParameters()) - 8.0f + 1.0f);
		    // * powf((float)modules[l][0]->getNParameters(), -.5f);
	}
	if (useInMLP) inMLPmutationProbability = baseMutationProbability /
		(log2f(powf(2.0f, 8.0f) + (float)inMLPs[0]->getNParameters()) - 8.0f + 1.0f);

	if (useOutMLP) outMLPmutationProbability = baseMutationProbability /
		(log2f(powf(2.0f, 8.0f) + (float)outMLPs[0]->getNParameters()) - 8.0f + 1.0f);


	nNodesPerNetwork = 0;
	int inputArraySize = 0;
	int destinationArraySize = 0;
	nModulesPerNetworkLayer = std::make_unique<int[]>(nLayers);
	nModulesPerNetworkLayer[0] = 1;
	for (int l = 0; l < nLayers; l++)
	{
		int nc = nChildrenPerLayer[l];
		if (l < nLayers - 1) nModulesPerNetworkLayer[l + 1] = nc * nModulesPerNetworkLayer[l];

		int cIs = nc == 0 ? 0 : inSizes[l + 1];
		destinationArraySize += nModulesPerNetworkLayer[l] *
			(outSizes[l] + cIs * nc);

		int cOs = nc == 0 ? 0 : outSizes[l + 1];
		inputArraySize += nModulesPerNetworkLayer[l] *
			(inSizes[l] + cOs * nc);

		nNodesPerNetwork += nModulesPerNetworkLayer[l];
	}



	Network::destinationArraySize = destinationArraySize;
	Network::inputArraySize = inputArraySize;
	Network::inS = inSizes;
	Network::outS = outSizes;
	Network::nC = nChildrenPerLayer;
	Network::nLayers = nLayers;

	networks = new Network*[nSpecimens];
	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = new Network(nTrialsPerGroup);
		createPhenotype(networks[i]);
	}

#ifdef NOCUDA
	GPUpreallocForNetworks();
#endif

	Network::mats_CUDA = matrices_LayerDestinationType_CUDA;
	Network::vecs_CUDA = vectors_LayerDestinationType_CUDA;
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

	if (useInMLP)
	{
		for (int j = 0; j < nInMLPs; j++) {
			delete inMLPphylogeneticTree[j];
			delete inMLPs[j];
		}
		delete[] inMLPphylogeneticTree;
		delete[] inMLPs;

		for (int i = 0; i < toBeDestroyedInMLPs.size(); i++) {
			delete toBeDestroyedInMLPs[i];
		}
	}

	if (useOutMLP)
	{
		for (int j = 0; j < nOutMLPs; j++) {
			delete outMLPphylogeneticTree[j];
			delete outMLPs[j];
		}
		delete[] outMLPphylogeneticTree;
		delete[] outMLPs;

		for (int i = 0; i < toBeDestroyedOutMLPs.size(); i++) {
			delete toBeDestroyedOutMLPs[i];
		}
	}


	for (int i = 0; i < nSpecimens; i) {
		delete networks[i];
	}
	delete[] networks;

	delete[] groupFitnesses;

	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		delete toBeDestroyedModules[i];
	}

#ifndef NOCUDA
	GPUdeallocForNetworks();
#endif
}


void Population::zeroModulesAccumulators() {
	for (int l = 0; l < nLayers; l++) {
		for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
			modules[l][i]->tempFitnessAccumulator = 0.0f;
			modules[l][i]->nTempFitnessAccumulations = 0;
		}
	}
}


void Population::zeroMLPsaccumulators() {
	if (useInMLP) {
		for (int i = 0; i < nInMLPs; i++) {
			inMLPs[i]->tempFitnessAccumulator = 0.0f;
			inMLPs[i]->nTempFitnessAccumulations = 0;
		}
	}
	if (useOutMLP) {
		for (int i = 0; i < nOutMLPs; i++) {
			outMLPs[i]->tempFitnessAccumulator = 0.0f;
			outMLPs[i]->nTempFitnessAccumulations = 0;
		}
	}
}


void Population::evolve(int nSteps) 
{
	// increasing these parameters refines the fitness computations, at a linear compute cost.
	// (linear in nShuffles * nNetworksSelectionSteps)

	// how many different groups a network will be part of between 2 network selection steps.
	const int nNetworksShuffles = 3;

	// how many selection steps there are for networks between two module selections
	const int nNetworksCyclesPerModuleCycle = 3;


	for (int i = 0; i < nSteps; i++) 
	{
		
		zeroMLPsaccumulators();
		zeroModulesAccumulators();

		for (int j = 0; j < nNetworksCyclesPerModuleCycle; j++)
		{

#ifdef NO_GROUP
			bool log = (j == 0);
			if (log) std::cout << "Step " << i << " ";

			evaluateNetsIndividually(log); 
#else

			// Sort networks by age
			auto f = [](Network* n1, Network* n2) {return n1->nExperiencedTrials > n2->nExperiencedTrials; };
			std::sort(networks, networks + nSpecimens, f);


			for (int k = 0; k < nNetworksShuffles; k++)
			{
				// Shuffle networks inside each age group
				for (int i = 0; i < groupTrial->nAgents; i++) {
					std::shuffle(networks + i * nGroups, networks + (i + 1) * nGroups, generator);
				}

				bool log = (k + j) == 0; // TODO monitor the whole process
				if (log) std::cout << "Step " << i << " ";

				evaluateGroups(log); 
			}
#endif

			replaceNetworks();
		}

		replaceMLPs();
		deleteUnusedMLPs();

		replaceModules();
		deleteUnusedModules();
	} 
}


void Population::evaluateNetsIndividually(bool log) 
{
	// nSpecimens = nGroups when this function is called.

	for (int i = 0; i < nSpecimens; i++) 
	{
		for (int j = 0; j < nTrialsPerGroup; j++)
		{
			networks[i]->preTrialReset();
			groupTrial->innerTrial->reset(false);
			if (networks[i]->outMLP->type->nLayers != 2) {
				__debugbreak(); // TODO remove (track first why type is incorrect / dealocated. (yet sizes is correct))
			}
			while (!groupTrial->innerTrial->isTrialOver)
			{
				networks[i]->step(groupTrial->innerTrial->observations.data());
				groupTrial->innerTrial->step(networks[i]->getOutput());
			}
			groupFitnesses[j * nSpecimens + i] = groupTrial->innerTrial->score;
		}
	}

	// monitoring
	if (log) {
		float avgF = 0.0f;
		float maxF = -100000.0f;
		float* avgNetF = new float[nSpecimens];
		std::fill(avgNetF, avgNetF + nSpecimens, 0.0f);
		for (int i = 0; i < nTrialsPerGroup * nSpecimens; i++)
		{
			if (groupFitnesses[i] > maxF) maxF = groupFitnesses[i];
			avgF += groupFitnesses[i];
			avgNetF[i % nSpecimens] += groupFitnesses[i];
		}
		avgF /= (float)(nTrialsPerGroup * nSpecimens);
		float maxAvgNetF = -100000.0f;
		for (int i = 0; i < nGroups; i++)
		{
			if (maxAvgNetF < avgNetF[i]) maxAvgNetF = avgNetF[i];
		}
		maxAvgNetF /= nTrialsPerGroup;
		delete[] avgNetF;

		// scores are per group.
		std::cout << " Over networks, avg avg score: " << avgF << ", best avg score " << maxAvgNetF << ", best score : " << maxF << std::endl;
	}

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
		if (useInMLP) {
			networks[i]->inMLP->type->tempFitnessAccumulator += f;
			networks[i]->inMLP->type->nTempFitnessAccumulations++;
		}
		if (useOutMLP) {
			networks[i]->outMLP->type->tempFitnessAccumulator += f;
			networks[i]->outMLP->type->nTempFitnessAccumulations++;
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

	MLP_P* inMLP = nullptr;
	if (useInMLP) {
		inMLP = new MLP_P(inMLPs[INT_0X(nInMLPs)]);
	}

	MLP_P* outMLP = nullptr;
	if (useOutMLP) {
		outMLP = new MLP_P(outMLPs[INT_0X(nOutMLPs)]);
	}

	// ownership of the pointers is given to the network.
	net->createPhenotype(genotype, inMLP, outMLP);
}


Node_G* Population::createModuleChild(PhylogeneticNode* primaryParent, int moduleLayer) {

	if (maxNParents == 1) {
		return new Node_G(modules[moduleLayer][primaryParent->modulePosition]);
	}

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

		for (int depth = consanguinityDistance; depth < MAX_PHYLOGENETIC_DEPTH; depth++) {

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


MLP_G* Population::createMLPChild(PhylogeneticNode* primaryParent, bool inOrOut)
{
	MLP_G** MLPs = inOrOut ? inMLPs : outMLPs;

	if (maxNParents == 1) {
		return new MLP_G(MLPs[primaryParent->modulePosition]);
	}

	std::vector<int> parents;
	parents.push_back(primaryParent->modulePosition);

	std::vector<float> rawWeights;
	rawWeights.resize(maxNParents);

	// fill parents, and weights are initialized with a function of the phylogenetic distance.
	{
		bool listFilled = false;

		auto distanceValue = [](float d)
		{
			return powf(1.0f + d, -0.6f);
		};

		rawWeights[0] = 1.0f; // = distanceValue(0)

		for (int depth = consanguinityDistance; depth < MAX_PHYLOGENETIC_DEPTH; depth++) {

			float w = distanceValue((float)depth);
			std::fill(rawWeights.begin() + (int)parents.size(), rawWeights.end(), w);

			primaryParent->fillList(parents, 0, depth, nullptr, maxNParents);

			// parents.size() == nParents can happen if there were exactly as many parents 
			// as there was room left in the array. This does not set listFilled = true.
			if (listFilled || parents.size() == maxNParents) break;
		}
	}

	if (parents.size() == 1) { // There were no close enough relatives.
		return new MLP_G(MLPs[primaryParent->modulePosition]);
	}

	float f0 = MLPs[primaryParent->modulePosition]->lifetimeFitness;

	std::vector<MLP_G*> parentNodes;

#ifdef  SPARSE_MUTATION_AND_COMBINATIONS
	parentNodes.push_back(MLPs[parents[0]]);
	for (int i = 1; i < parents.size(); i++) {
		if (MLPs[parents[i]]->lifetimeFitness > f0) {
			parentNodes.push_back(MLPs[parents[i]]);
		}
	}
	if (parentNodes.size() == 1) { // There were no close enough relatives with a better fitness
		return new MLP_G(MLPs[primaryParent->modulePosition]);
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


	return MLP_G::combine(parentNodes.data(), rawWeights.data(), (int)parentNodes.size());
}


void Population::replaceMLPs()
{

	if (useInMLP) {

		for (int i = 0; i < nInMLPs; i++) {
			MLP_G* m = inMLPs[i];
			if (m->nTempFitnessAccumulations == 0) continue;
			m->tempFitnessAccumulator /= (float)m->nTempFitnessAccumulations;
			m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay + m->tempFitnessAccumulator;
		}

		// Compute roulette probabilities.
		{
			float invProbaSum = 0.0f;

			for (int i = 0; i < nInMLPs; i++) {
				float pRaw = inMLPs[i]->lifetimeFitness - 0.0f;

				// TODO either this or a (uniform ?) probability over the X best
				// percentile. But that is yet another parameter.
				if (pRaw > 0) probabilities[i] = pRaw;


				else probabilities[i] = 0.0f;

				invProbaSum += probabilities[i];
			}

			invProbaSum = 1.0f / invProbaSum;

			probabilities[0] = probabilities[0] * invProbaSum;
			for (int i = 1; i < nInMLPs; i++) {
				probabilities[i] = probabilities[i - 1] + probabilities[i] * invProbaSum;
			}
		}

		int nReplacements = 0;

		for (int i = 0; i < nInMLPs; i++) {
			if (inMLPs[i]->lifetimeFitness < currentInMLPReplacementTreshold)
			{
				inMLPs[i]->isStillEvolved = false;
				toBeDestroyedInMLPs.push_back(inMLPs[i]);

				inMLPphylogeneticTree[i]->onModuleDestroyed();

				int parentID = binarySearch(probabilities.get(), UNIFORM_01, nInMLPs);

				inMLPs[i] = createMLPChild(inMLPphylogeneticTree[parentID], true);
				inMLPs[i]->mutate(inMLPmutationProbability);
				inMLPs[i]->isStillEvolved = true;

				inMLPphylogeneticTree[i] = new PhylogeneticNode(inMLPphylogeneticTree[parentID], i);

				nReplacements++;
			}
		}

		if ((float)nReplacements / (float)nInMLPs > inMLPReplacedFraction)
		{
			currentInMLPReplacementTreshold = std::max(-10.0f, currentInMLPReplacementTreshold / .8f);
		}
		else {
			currentInMLPReplacementTreshold = std::min(-.03f, currentInMLPReplacementTreshold * .8f);
		}
	}


	if (useOutMLP) {

		for (int i = 0; i < nOutMLPs; i++) {
			MLP_G* m = outMLPs[i];
			if (m->nTempFitnessAccumulations == 0) continue;
			m->tempFitnessAccumulator /= (float)m->nTempFitnessAccumulations;
			m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay + m->tempFitnessAccumulator;
		}

		// Compute roulette probabilities.
		{ 
			float invProbaSum = 0.0f;

			for (int i = 0; i < nOutMLPs; i++) {
				float pRaw = outMLPs[i]->lifetimeFitness - 0.0f;

				// TODO either this or a (uniform ?) probability over the X best
				// percentile. But that is yet another parameter.
				if (pRaw > 0) probabilities[i] = pRaw;


				else probabilities[i] = 0.0f;

				invProbaSum += probabilities[i];
			}

			invProbaSum = 1.0f / invProbaSum;

			probabilities[0] = probabilities[0] * invProbaSum;
			for (int i = 1; i < nOutMLPs; i++) {
				probabilities[i] = probabilities[i - 1] + probabilities[i] * invProbaSum;
			}
		}

		int nReplacements = 0;

		for (int i = 0; i < nOutMLPs; i++) {
			if (outMLPs[i]->lifetimeFitness < currentOutMLPReplacementTreshold)
			{
				outMLPs[i]->isStillEvolved = false;
				toBeDestroyedOutMLPs.push_back(outMLPs[i]);

				outMLPphylogeneticTree[i]->onModuleDestroyed();

				int parentID = binarySearch(probabilities.get(), UNIFORM_01, nOutMLPs);

				outMLPs[i] = createMLPChild(outMLPphylogeneticTree[parentID], false);
				outMLPs[i]->mutate(outMLPmutationProbability);
				outMLPs[i]->isStillEvolved = true;

				outMLPphylogeneticTree[i] = new PhylogeneticNode(outMLPphylogeneticTree[parentID], i);

				nReplacements++;
			}
		}

		if ((float)nReplacements / (float)nOutMLPs > outMLPReplacedFraction)
		{
			currentOutMLPReplacementTreshold = std::max(-10.0f, currentOutMLPReplacementTreshold / .8f);
		}
		else {
			currentOutMLPReplacementTreshold = std::min(-.03f, currentOutMLPReplacementTreshold * .8f);
		}
	}
}


void Population::deleteUnusedMLPs()
{

	if (useInMLP)
	{
		for (int i = 0; i < toBeDestroyedInMLPs.size(); i++) {
			if (toBeDestroyedInMLPs[i]->nUsesInNetworks == 0) {
				delete toBeDestroyedInMLPs[i];

				// For cleaning in the following line.
				toBeDestroyedInMLPs[i] = nullptr;
			}
		}
		auto eraseUnused = [](MLP_G* n) {
			return n == nullptr;
		};

		std::erase_if(toBeDestroyedInMLPs, eraseUnused);
	}


	if (useOutMLP)
	{
		for (int i = 0; i < toBeDestroyedOutMLPs.size(); i++) {
			if (toBeDestroyedOutMLPs[i]->nUsesInNetworks == 0) {
				delete toBeDestroyedOutMLPs[i];

				// For cleaning in the following line.
				toBeDestroyedOutMLPs[i] = nullptr;
			}
		}
		auto eraseUnused = [](MLP_G* n) {
			return n == nullptr;
		};

		std::erase_if(toBeDestroyedOutMLPs, eraseUnused);
	}
}


void Population::deleteUnusedModules()
{
	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		if (toBeDestroyedModules[i]->nUsesInNetworks == 0) {
			delete toBeDestroyedModules[i];

			// For cleaning in the following line.
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

		networks[i] = new Network(nTrialsPerGroup);
		createPhenotype(networks[i]);
	}


	// TODO .8 arbitrary
	if ((float)nReplacements / (float)nSpecimens > networkReplacedFraction)  
	{
		currentNetworkReplacementTreshold = std::max(-10.0f, currentNetworkReplacementTreshold / .8f); 
	}
	else {
		currentNetworkReplacementTreshold = std::min(-.03f, currentNetworkReplacementTreshold * .8f);
	}
}


void Population::replaceModules() {
	uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	for (int l = 0; l < nLayers; l++) {

		for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
			Node_G* m = modules[l][i];
			if (m->nTempFitnessAccumulations == 0) continue;
			m->tempFitnessAccumulator /= (float)m->nTempFitnessAccumulations;
			m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay + m->tempFitnessAccumulator;
		}
		

		// Compute roulette probabilities.
		{
			float invProbaSum = 0.0f;

			for (int i = 0; i < nEvolvedModulesPerLayer[l]; i++) {
				float pRaw = modules[l][i]->lifetimeFitness - 0.0f;

				// TODO either this or a (uniform ?) probability over the X best
				// percentile. But that is yet another parameter.
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
				modules[l][i]->isStillEvolved = false;
				toBeDestroyedModules.push_back(modules[l][i]);

				phylogeneticTrees[l][i]->onModuleDestroyed();

				int parentID = binarySearch(probabilities.get(), UNIFORM_01, nEvolvedModulesPerLayer[l]);

				modules[l][i] = createModuleChild(phylogeneticTrees[l][parentID], l);
				modules[l][i]->mutate(mutationsProbabilitiesPerLayer[l]);
				modules[l][i]->isStillEvolved = true;

				phylogeneticTrees[l][i] = new PhylogeneticNode(phylogeneticTrees[l][parentID], i);

				nReplacements++;
			}
		}
		//std::cout << (float)nReplacements / (float)nEvolvedModulesPerLayer[l] << " ";
		if ((float)nReplacements / (float)nEvolvedModulesPerLayer[l] > moduleReplacedFractions[l])
		{
			currentModuleReplacementTreshold[l] = std::max(-10.0f, currentModuleReplacementTreshold[l] / .8f);
		}
		else {
			currentModuleReplacementTreshold[l] = std::min(-.03f, currentModuleReplacementTreshold[l] * .8f);
		}
	}
	//std::cout << std::endl;

	uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//std::cout << "Modules creation took " << stop - start << " ms." << std::endl;
}


void Population::GPUdeallocForNetworks() {
	for (int l = 0; l < nLayers; l++)
	{
		for (int dest = 0; dest < 3; dest++) {
			for (int i = 0; i < N_DYNAMIC_VECTORS; i++) { // TODO  N_DYNAMIC_VECTORS is not the correct quantity !!!
				cudaFree(vectors_LayerDestinationType_CUDA[l][dest][i]);
			}

			for (int i = 0; i < N_DYNAMIC_MATRICES; i++) {
				cudaFree(matrices_LayerDestinationType_CUDA[l][dest][i]);
			}
		}
	}
}


void Population::GPUpreallocForNetworks() 
{
	
	vectors_LayerDestinationType_CUDA = new float***[nLayers];
	matrices_LayerDestinationType_CUDA = new float***[nLayers];

	for (int l = 0; l < nLayers; l++) 
	{
		int totalNModulesAtLayer = groupTrial->nAgents * nModulesPerNetworkLayer[l];


		vectors_LayerDestinationType_CUDA[l] = new float**[3];
		matrices_LayerDestinationType_CUDA[l] = new float**[3];

	
		InternalConnexion_G* co[3] = {
			&modules[l][0]->toChildren,
			&modules[l][0]->toOutput,
		};

		for (int dest = 0; dest < 3; dest++) {

			vectors_LayerDestinationType_CUDA[l][dest] = new float*[N_DYNAMIC_VECTORS];
			int vecSize = co[dest]->nRows;
			for (int i = 0; i < N_DYNAMIC_VECTORS; i++) {
				cudaMalloc(
					&vectors_LayerDestinationType_CUDA[l][dest][i],
					sizeof(float) * vecSize * totalNModulesAtLayer
				);
			}


			matrices_LayerDestinationType_CUDA[l][dest] = new float*[N_DYNAMIC_MATRICES];
			int matSize = co[dest]->nRows * co[dest]->nColumns;
			for (int i = 0; i < N_DYNAMIC_MATRICES; i++) {
				cudaMalloc(
					&matrices_LayerDestinationType_CUDA[l][dest][i],
					sizeof(float) * matSize * totalNModulesAtLayer
				);
			}
		}
	}
}
