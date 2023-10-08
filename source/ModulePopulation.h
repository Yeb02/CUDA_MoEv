#pragma once

#include <algorithm>

#include "VirtualClasses.h"
#include "Random.h"

#include "Node_G.h"


struct PhylogeneticNode
{
	// initialized in main.cpp before main().
	static int maxPhylogeneticDepth;

	// The PNode associated with the module that was "this"'s module's primary parent. nullptr if this
	// is the first generation or the parent was too deep.
	PhylogeneticNode* parent;

	// the position of the associated module in the module population. Is -1 if the associated module
	// has been deleted or is in the toBeDeleted list.
	int modulePosition;

	// pointers towards all the children that still exist. 
	std::vector<PhylogeneticNode*> children;

	// if depth >= MAX_MATING_DEPTH, "this" is deleted. depth is the distance to the closest descendant
	// that is associated with a module that is still in the module array. So 0 if modulePosition != -1 
	int depth;

	PhylogeneticNode() {};

	PhylogeneticNode(PhylogeneticNode* parent, int _modulePosition) :
		modulePosition(_modulePosition), parent(parent)
	{
		depth = 0;

		// parent is nullptr at startup for the very first generation.
		if (parent!=nullptr) parent->children.push_back(this);

		// no need to update the parents depth as it is already 0, since only a 
		// phylogenetic node with an active associated module can be a parent.
	};

	// Does not actually deallocate this, it must be handled outside. Deletion happens in
	// updateDepth(). Parent depth is not updated here because the only place this function 
	// is called is in updateDepth which already handles it.
	void destroy()
	{
		for (int i = 0; i < children.size(); i++) {
			children[i]->parent = nullptr;
		}

		// The following block is equivalent to 
		// remove(parent->children.begin(), parent->children.end(), this);
		// but much more efficient.
		if (parent != nullptr) {
			int s = (int)parent->children.size() - 1;
			for (int i = 0; i < s; i++) {
				if (parent->children[i] == this) {
					parent->children[i] = parent->children[s];
					break;
				}
			}
			parent->children.pop_back();
		}
	}

	// called when one of the children of this has its depth updated to d0.
	// (updated means increased, as depth can only increase !)
	// To be called with d0 = -1 if this's module has just been removed from the 
	// population and depth was previously 0.
	void updateDepth(int d0)
	{
		if (
			modulePosition != -1 || // this's module is alive, so depth stays at 0
			d0 > depth-1)           // the child whose depth increased was not one of the shallowest children
		{
			return;
		}

		int d1 = depth + 1;
		int minD = maxPhylogeneticDepth;
		for (int i = 0; i < children.size(); i++)
		{
			if (minD > children[i]->depth) minD = children[i]->depth;
		}
		depth = minD + 1;

		if (depth >= maxPhylogeneticDepth) 
		{ 
			if (parent != nullptr) parent->updateDepth(depth);
			destroy();
			delete this; // careful.
			return;
		}

		// the child node whose depth has increased was one of the !!several!! shallowest children,
		// and this's depth has therefore not changed. Parent's depth update is then not needed.
		if (depth == d1 - 1) return;

		if (parent != nullptr) parent->updateDepth(depth);
	}

	void onModuleDestroyed()
	{
		modulePosition = -1;
		updateDepth(-1);
	}

	// returns true if the list is full and we should exit the recursion, false otherwise.
	bool fillList(std::vector<int>& list, int currentDistance, int targetDistance, PhylogeneticNode* prevNode, int maxListSize) {
		if (currentDistance == targetDistance)
		{
			if (modulePosition != -1)
			{
				list.push_back(modulePosition);
			}
			return list.size() == maxListSize;
		}

		if (parent != prevNode && parent != nullptr) {
			if (parent->fillList(list, currentDistance + 1, targetDistance, this, maxListSize)) {
				return true;
			}
		}

		for (int i = 0; i < children.size(); i++) {
			if (children[i] == prevNode) continue;
			if (children[i]->fillList(list, currentDistance + 1, targetDistance, this, maxListSize)) {
				return true;
			}
		}
		return false;
	}
};


struct ModulePopulationParameters 
{
	// for all modules of this population, lifetimeFitness(t) = lifetimeFitness(t-1) * accumulatedFitnessDecay + fitness(t)
	float accumulatedFitnessDecay;

	// How many modules are evolved by this population
	int nModules;

	// In [0, 1]. All mutation probabilities are proportional to this value.
	float baseMutationProbability;

	// Positive integer value. Specimens whose phenotypic distance to the primary parent are below it
	// are not used for combination. MUST BE >= 1
	int consanguinityDistance;

	// Minimum at 1. If = 1, mutated clone of the parent. No explicit maximum, but bounded by nSpecimens, 
	// and maxPhylogeneticDepth implicitly. Cost O( n log(n) ).
	int maxNParents;

	// Maximum depth of the phylogenetic tree. This means that all pairs of modules used in a combination cannot be
	// be further away genetically than MAX_PHYLOGENETIC_DEPTH combinations and mutations. 
	// MUST BE >= 1
	int maxPhylogeneticDepth;

	// target fraction of the modules that is replaced at the end of a module cycle.
	float moduleReplacedFraction;

	// fraction of the population of modules that are used as parents.
	float moduleElitePercentile;
};


template <class Module> // how to enforce (and inform intellisense) that the type is derived from imodule ? TODO
class ModulePopulation
{


private:
	// parameters set once at construction. More details in SystemParameters:
	int maxNParents;
	int consanguinityDistance;
	int maxPhylogeneticDepth;
	float moduleReplacedFraction;
	float moduleElitePercentile;
	float accumulatedFitnessDecay;
	float mutationsProbability;
	int nModules;

	// biases sampling towards modules that have been sampled the least, relative to their age.
	std::unique_ptr<float[]> samplingProbabilities;

	// how many times each module has been sampled during its lifetime
	std::unique_ptr<int[]> nSamplings;


	// lifetime fitness threshold for modules.Dynamic quantity, adjusted after each replacement step
	// to match the actual replaced fraction with moduleReplacedFraction more closely at the next step.
	float currentModuleReplacementTreshold;

	// The list of alive modules. 
	std::vector<Module*> modules;

	// The list of modules that have been removed from the "modules" array but are still in use in some Agents.
	// Once all those agents have been destroyed, the module is destroyed.
	std::vector<Module*> toBeDestroyedModules;

	// The leaves of the phylogenetic tree.
	std::vector<PhylogeneticNode*> phylogeneticTree;



public:

	ModulePopulation(ModulePopulationParameters& p, IModuleFixedParameters* mParams) {
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
			modules[j] = (Module*) new Node_G(*(static_cast<Node_GFixedParameters*>(mParams))); //ugly i know
			phylogeneticTree[j] = new PhylogeneticNode(nullptr, j);
		}



		mutationsProbability = p.baseMutationProbability
			/ log2f((float)modules[0]->getNParameters()); // TODO
		// / (log2f(powf(2.0f, 8.0f) + (float)modules[0]->getNParameters()) - 8.0f + 1.0f);
		// * powf((float)modules[0]->getNParameters(), -.5f);
	}


	~ModulePopulation() {
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
	Module* createModuleChild(PhylogeneticNode* primaryParent) {
		if (maxNParents == 1) {
			return new Module(modules[primaryParent->modulePosition]);
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
			return new Module(modules[primaryParent->modulePosition]);
		}

		float f0 = modules[primaryParent->modulePosition]->lifetimeFitness;

		std::vector<Module*> parentNodes;

#ifdef  SPARSE_MUTATION_AND_COMBINATIONS
		parentNodes.push_back(modules[parents[0]]);
		for (int i = 1; i < parents.size(); i++) {
			if (modules[parents[i]]->lifetimeFitness > f0) {
				parentNodes.push_back(modules[parents[i]]);
			}
		}
		if (parentNodes.size() == 1) { // There were no close enough relatives with a better fitness
			return new Module(modules[primaryParent->modulePosition]);
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


		return Module::combine(parentNodes.data(), rawWeights.data(), (int)parentNodes.size());
	}


	Module* sample()
	{
		int id = binarySearch(samplingProbabilities.get(), UNIFORM_01, nModules);

		Module* m = modules[id];

		nSamplings[id]++;

		modules[id]->nUsesInAgents++;

		return modules[id];
	}


	void replaceModules() 
	{
		for (int i = 0; i < nModules; i++) {
			Module* m = modules[i];

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
			currentModuleReplacementTreshold =  currentModuleReplacementTreshold / .8f;
		}
		else {
			currentModuleReplacementTreshold = currentModuleReplacementTreshold * .8f;
		}



		// order does not matter:
		updateSamplingArrays();
		deleteUnusedModules();
		zeroModulesAccumulators();
	}

private:

	void updateSamplingArrays () 
	{

		float normalizer = 0.0f;
		int nOld = 0;
		for (int i = 0; i < nModules; i++)
		{
			if (modules[i]->age > 0) [[likely]]
			{
				nOld++;
				samplingProbabilities[i] = expf(-1.0f* (float)nSamplings[i] / (float)modules[i]->age); // used as temporary storage
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

	void deleteUnusedModules() 
	{
		for (int i = 0; i < toBeDestroyedModules.size(); i++) {
			if (toBeDestroyedModules[i]->nUsesInAgents == 0) {
				delete toBeDestroyedModules[i];

				// For cleaning in the following line.
				toBeDestroyedModules[i] = nullptr;
			}
		}
		auto eraseUnused = [](Module* n) {
			return n == nullptr;
		};

		std::erase_if(toBeDestroyedModules, eraseUnused);
	}

	void zeroModulesAccumulators() {
		for (int i = 0; i < nModules; i++) {
			modules[i]->tempFitnessAccumulator = 0.0f;
			modules[i]->nTempFitnessAccumulations = 0;
		}
	}

};

