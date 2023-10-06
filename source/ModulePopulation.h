#pragma once

#include <algorithm>

#include "VirtualClasses.h"
#include "Random.h"

#include "Node_G.h"


struct PhylogeneticNode
{
	static int maxPhylogeneticDepth;

	// The PNode associated with the module that was "this"'s module's primary parent. nullptr if this
	// is the first generation or the parent was too deep.
	PhylogeneticNode* parent;

	// the position of the associated module in the module population. Is -1 if the associated module
	// has been deleted or is in the toBeDeleted list.
	int modulePosition;

	// pointers towards all the children that still exist.
	std::vector<PhylogeneticNode*> children;

	// if depth >= MAX_MATING_DEPTH, "this" is deleted. depth is the distance to the closest child (indirect)
	// that is associated with a module that is still in the module array.
	int depth;

	PhylogeneticNode() {};

	PhylogeneticNode(PhylogeneticNode* parent, int _modulePosition) :
		modulePosition(_modulePosition), parent(parent)
	{
		depth = 0;
		if (parent != nullptr) parent->children.push_back(this);
		// no need to update the parents depth as it is already 0, since only a 
		// phylogenetic node with an active associated module
	};

	// Does not actually desallocate this, it must be handled outside !
	void destroy()
	{
		for (int i = 0; i < children.size(); i++) {
			children[i]->parent = nullptr;
		}

		if (parent != nullptr) {

			// The following block is equivalent to 
			// remove(parent->children.begin(), parent->children.end(), this);
			// but much more efficient.

			int s = (int)parent->children.size() - 1;
			for (int i = 0; i < s; i++) {
				if (parent->children[i] == this) {
					parent->children[i] = parent->children[s];
					break;
				}
			}
			parent->children.pop_back();



			parent->updateDepth(-1);
		}
	}

	void updateDepth(int d0)
	{
		if (
			modulePosition != -1 || // module is alive
			d0 > depth)            // the child whose depth increased was not one of the shallowest
		{
			return;
		}

		int d1 = depth + 1;
		int minD = maxPhylogeneticDepth - 1;
		for (int i = 0; i < children.size(); i++)
		{
			if (minD > children[i]->depth) minD = children[i]->depth;
		}
		depth = minD + 1;

		if (depth == maxPhylogeneticDepth) {
			destroy();
			delete this; // careful.
			return;
		}

		// the child node whose depth has increased was one of the !!several!! shallowest children,
		// and this's depth has therefore not changed.
		if (depth = d1 - 1) return;

		if (parent != nullptr) parent->updateDepth(d1);
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

int PhylogeneticNode::maxPhylogeneticDepth = 0;


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

template <class Module> // how to make sure the type is derived from imodule ?
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

	// how many times each module has been sampled, divided by its age.
	std::unique_ptr<float[]> avgNumberOfSamplings;


	// lifetime fitness threshold for modules.Dynamic quantity, adjusted after each replacement step
	// to match the actual replaced fraction with moduleReplacedFraction more closely at the next step.
	float currentModuleReplacementTreshold;

	// The list of alive modules. 
	std::unique_ptr<Module* []> modules;

	// The list of modules that have been removed from the "modules" array but are still in use in some Agents.
	// Once all those agents have been destroyed, the module is destroyed.
	std::vector<Module*> toBeDestroyedModules;

	// The leaves of the phylogenetic tree.
	std::unique_ptr<PhylogeneticNode*[]> phylogeneticTree;



public:

	ModulePopulation(ModulePopulationParameters& p, IModuleFixedParameters* mParams) {
		maxNParents = p.maxNParents;
		consanguinityDistance = p.consanguinityDistance;
		maxPhylogeneticDepth = p.maxPhylogeneticDepth;

		PhylogeneticNode::maxPhylogeneticDepth = maxPhylogeneticDepth;

		moduleReplacedFraction = p.moduleReplacedFraction;
		moduleElitePercentile = p.moduleElitePercentile;
		accumulatedFitnessDecay = p.accumulatedFitnessDecay;
		mutationsProbability = p.mutationsProbability;
		nModules = p.nModules;

		// TODO -.7 arbitrary. As it only affects initialization, it can stay here, but
		// work of satisfactory values must be done.
		currentModuleReplacementTreshold = -.7f; // <0


		samplingProbabilities = std::make_unique<float[]>(nModules);
		std::fill(&samplingProbabilities[0], &samplingProbabilities[nModules], 1.0f/(float)nModules)
		avgNumberOfSamplings = std::make_unique<float[]>(nModules);
		std::fill(&avgNumberOfSamplings[0], &avgNumberOfSamplings[nModules], 0.0f)


		modules = std::make_unique<Module*[]>(nModules);
		phylogeneticTree = std::make_unique<PhylogeneticNode *[]>(nModules);

		
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
		int id = binarySearch(samplingProbabilities, UNIFORM_01);

		Module* m = modules[id];

		if (m->age > 0) {
			avgNumberOfSamplings[id] = (avgNumberOfSamplings[id] * (float)m->age + 1.0f) / (float)m->age;
		}
		else {
			avgNumberOfSamplings[id]++;
		}
		

		modules[id]->nUsesInAgents++;

		return modules[id];
	}


	void replaceModules() 
	{
		for (int i = 0; i < nModules; i++) {
			Module* m = modules[i];
			if (m->nTempFitnessAccumulations == 0) continue;
			m->tempFitnessAccumulator /= (float)m->nTempFitnessAccumulations;
			m->lifetimeFitness = m->lifetimeFitness * accumulatedFitnessDecay + m->tempFitnessAccumulator;
			m->age++;
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
			if (modules[i]->lifetimeFitness < currentModuleReplacementTreshold)
			{
				modules[i]->isStillEvolved = false;
				toBeDestroyedModules.push_back(modules[i]);

				phylogeneticTree[i]->onModuleDestroyed();

				int parentID = positions[INT_0X(nEliteModules)];

				modules[i] = createModuleChild(phylogeneticTree[parentID]);
				modules[i]->mutate(mutationsProbability);
				
				avgNumberOfSamplings[i] = 0.0f;

				phylogeneticTree[i] = new PhylogeneticNode(phylogeneticTree[parentID], i);

				nReplacements++;
			}
		}
		//std::cout << (float)nReplacements / (float)nModules << std::endl;
		if ((float)nReplacements / (float)nModules > moduleReplacedFraction)
		{
			currentModuleReplacementTreshold = std::max(-10.0f, currentModuleReplacementTreshold / .8f);
		}
		else {
			currentModuleReplacementTreshold = std::min(-.03f, currentModuleReplacementTreshold * .8f);
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
			if (modules[i]->age != 0) [[likely]]
			{
				nOld++;
				normalizer += avgNumberOfSamplings[i];
			}
		}

		normalizer = (float)nOld / (normalizer * (float)nModules);
		float newBornProba = 1.0f / (float)nModules;

		samplingProbabilities[0] = modules[0]->age == 0 ? newBornProba : samplingProbabilities[0] * normalizer;
		for (int i = 1; i < nModules; i++)
		{
			if (modules[i]->age != 0) [[likely]]
			{
				samplingProbabilities[i] = samplingProbabilities[i - 1] + avgNumberOfSamplings[i] * normalizer;
			}
			else [[unlikely]]
			{
				samplingProbabilities[i] = samplingProbabilities[i - 1] + newBornProba;
			}
		}

		for (int i = 0; i < nModules; i++)
		{
			avgNumberOfSamplings[i] *= (float)modules[i]->age / ((float)modules[i]->age + 1); 
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
