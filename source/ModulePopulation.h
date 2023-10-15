#pragma once

#include <algorithm>

#include "MoEvCore.h"

#ifdef PREDICTIVE_CODING
#include "PC_Node_G.h"
#else
#include "HebbianNode_G.h"
#endif



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
	std::vector<MODULE*> modules;

	// The list of modules that have been removed from the "modules" array but are still in use in some Agents.
	// Once all those agents have been destroyed, the module is destroyed.
	std::vector<MODULE*> toBeDestroyedModules;

	// The leaves of the phylogenetic tree.
	std::vector<PhylogeneticNode*> phylogeneticTree;



public:

	ModulePopulation(ModulePopulationParameters& p, MODULE_PARAMETERS& mParams);

	~ModulePopulation();


	// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
	MODULE* createModuleChild(PhylogeneticNode* primaryParent);

	MODULE* sample();


	void replaceModules();

private:

	void updateSamplingArrays();

	void deleteUnusedModules();

	void zeroModulesAccumulators();

};


