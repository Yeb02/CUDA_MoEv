#pragma once

#include "ModulePopulation.h"

ModulePopulation::ModulePopulation(ModulePopulationParameters& p, MODULE_PARAMETERS& mParams) 
{
	maxNParents = p.maxNParents;
	consanguinityDistance = p.consanguinityDistance;
	maxPhylogeneticDepth = p.maxPhylogeneticDepth;

	PhylogeneticNode::maxPhylogeneticDepth = maxPhylogeneticDepth;

	moduleReplacedFraction = p.moduleReplacedFraction;
	moduleElitePercentile = p.moduleElitePercentile;
	accumulatedFitnessDecay = p.accumulatedFitnessDecay;
	nModules = p.nModules;


	currentModuleReplacementTreshold = -(1.0f - accumulatedFitnessDecay) * moduleReplacedFraction * 2.0f; // <0


	samplingProbabilities = std::make_unique<float[]>(nModules);
	std::fill(&samplingProbabilities[0], &samplingProbabilities[nModules], 1.0f / (float)nModules);
	nSamplings = std::make_unique<int[]>(nModules);
	std::fill(&nSamplings[0], &nSamplings[nModules], 0);


	modules.resize(nModules);
	phylogeneticTree.resize(nModules);


	for (int j = 0; j < nModules; j++)
	{
		modules[j] = new MODULE(mParams);
		phylogeneticTree[j] = new PhylogeneticNode(nullptr, j);
	}



	mutationsProbability = p.baseMutationProbability
		/ log2f((float)modules[0]->getNParameters()); // TODO
	// / (log2f(powf(2.0f, 8.0f) + (float)modules[0]->getNParameters()) - 8.0f + 1.0f);
	// * powf((float)modules[0]->getNParameters(), -.5f);
}


ModulePopulation::~ModulePopulation() {
	for (int j = 0; j < nModules; j++)
	{
		delete modules[j];
		delete phylogeneticTree[j];
	}

	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		delete toBeDestroyedModules[i];
	}
}


// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
MODULE* ModulePopulation::createModuleChild(PhylogeneticNode* primaryParent) {
	if (maxNParents == 1) {
		return new MODULE(modules[primaryParent->modulePosition]);
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

		for (int depth = consanguinityDistance; depth < maxPhylogeneticDepth; depth++) {

			float w = distanceValue((float)depth);
			std::fill(rawWeights.begin() + (int)parents.size(), rawWeights.end(), w);

			primaryParent->fillList(parents, 0, depth, nullptr, maxNParents);

			// parents.size() == nParents can happen if there were exactly as many parents 
			// as there was room left in the array. This does not set listFilled = true.
			if (listFilled || parents.size() == maxNParents) break;
		}
	}

	if (parents.size() == 1) { // There were no close enough relatives.
		return new MODULE(modules[primaryParent->modulePosition]);
	}

	float f0 = modules[primaryParent->modulePosition]->lifetimeFitness;

	std::vector<MODULE*> parentNodes;

#ifdef  SPARSE_COMBINATIONS
	parentNodes.push_back(modules[parents[0]]);
	for (int i = 1; i < parents.size(); i++) {
		if (modules[parents[i]]->lifetimeFitness > f0) {
			parentNodes.push_back(modules[parents[i]]);
		}
	}
	if (parentNodes.size() == 1) { // There were no close enough relatives with a better fitness
		return new MODULE(modules[primaryParent->modulePosition]);
	}
#else

	parentHebbianNetworks.resize(parents.size());
	for (int i = 0; i < parents.size(); i++) {
		parentHebbianNetworks[i] = networks[parents[i]];
	}

	// weights are an increasing function of the relative fitness.
	{
		rawWeights[0] = 1.0f;
		for (int i = 1; i < parents.size(); i++) {
			rawWeights[i] *= (fitnesses[parents[i]] - f0); // TODO  adapt to MoEv
		}
	}
#endif


	return MODULE::combine(parentNodes.data(), rawWeights.data(), (int)parentNodes.size());
}


MODULE* ModulePopulation::sample()
{
	int id = binarySearch(samplingProbabilities.get(), UNIFORM_01, nModules);

	MODULE* m = modules[id];

	nSamplings[id]++;

	modules[id]->nUsesInAgents++;

	return modules[id];
}


void ModulePopulation::replaceModules()
{
	for (int i = 0; i < nModules; i++) {
		MODULE* m = modules[i];

		m->age++;

		if (m->nTempFitnessAccumulations == 0) continue;


		m->tempFitnessAccumulator /= (float)m->nTempFitnessAccumulations;

		// the exponential avg starts at 0 for "newborns"
		m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay +
			(1.0f - accumulatedFitnessDecay) * m->tempFitnessAccumulator;

	}


	std::vector<int> positions(nModules);
	for (int i = 0; i < nModules; i++) {
		positions[i] = i;
	}
	// sort positions by descending lifetime fitness.
	std::sort(positions.begin(), positions.end(), [&](int a, int b) -> bool
		{
			return modules[a]->lifetimeFitness > modules[b]->lifetimeFitness;
		}
	);

	int nReplacements = 0;
	int nEliteModules = (int)((float)nModules * moduleElitePercentile);
	for (int i = 0; i < nModules; i++) {
		// If the module is fitter than the threshold, continue to next module.
		if (modules[i]->lifetimeFitness >= currentModuleReplacementTreshold) continue;

#ifdef YOUNG_AGE_BONUS
		// The younger a module is, the higher its probability of being randomly saved.
		if (powf(1.4f, (float)(-1 - modules[i]->age)) > UNIFORM_01) continue;
#endif

		modules[i]->isStillEvolved = false;
		toBeDestroyedModules.push_back(modules[i]);

		phylogeneticTree[i]->onModuleDestroyed();

		int parentID = positions[INT_0X(nEliteModules)];

		modules[i] = createModuleChild(phylogeneticTree[parentID]);
		modules[i]->mutate(mutationsProbability);

		nSamplings[i] = 0;

		phylogeneticTree[i] = new PhylogeneticNode(phylogeneticTree[parentID], i);

		nReplacements++;

	}
	//std::cout << (float)nReplacements / (float)nModules << std::endl;
	if ((float)nReplacements / (float)nModules > moduleReplacedFraction)
	{
		currentModuleReplacementTreshold = currentModuleReplacementTreshold / .8f;
	}
	else {
		currentModuleReplacementTreshold = currentModuleReplacementTreshold * .8f;
	}



	// order does not matter:
	updateSamplingArrays();
	deleteUnusedModules();
	zeroModulesAccumulators();
}



void ModulePopulation::updateSamplingArrays()
{

	float normalizer = 0.0f;
	int nOld = 0;
	for (int i = 0; i < nModules; i++)
	{
		if (modules[i]->age > 0) [[likely]]
		{
			nOld++;
			samplingProbabilities[i] = expf(-1.0f * (float)nSamplings[i] / (float)modules[i]->age); // used as temporary storage
			normalizer += samplingProbabilities[i];
		}
	}

	normalizer = (float)nOld / (.0000001f + normalizer * (float)nModules); // +epsilon because /0 at first step.
	float newBornProba = 1.0f / (float)nModules;

	samplingProbabilities[0] = modules[0]->age == 0 ? newBornProba : samplingProbabilities[0] * normalizer;
	for (int i = 1; i < nModules; i++)
	{
		if (modules[i]->age > 0) [[likely]]
		{
			samplingProbabilities[i] = samplingProbabilities[i - 1] + normalizer * samplingProbabilities[i];
		}
		else [[unlikely]]
		{
			samplingProbabilities[i] = samplingProbabilities[i - 1] + newBornProba;
		}
	}
}

void ModulePopulation::deleteUnusedModules()
{
	for (int i = 0; i < toBeDestroyedModules.size(); i++) {
		if (toBeDestroyedModules[i]->nUsesInAgents == 0) {
			delete toBeDestroyedModules[i];

			// For cleaning in the following line.
			toBeDestroyedModules[i] = nullptr;
		}
	}
	auto eraseUnused = [](MODULE* n) {
		return n == nullptr;
	};

	std::erase_if(toBeDestroyedModules, eraseUnused);
}

void ModulePopulation::zeroModulesAccumulators() {
	for (int i = 0; i < nModules; i++) {
		modules[i]->tempFitnessAccumulator = 0.0f;
		modules[i]->nTempFitnessAccumulations = 0;
	}
}