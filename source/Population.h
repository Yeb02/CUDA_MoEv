#pragma once

#include <memory>
#include <cmath>
#include <string>

#include "Network.h"
#include "Trial.h"
#include "Random.h"


const enum SCORE_BATCH_TRANSFORMATION { NONE = 0, NORMALIZE = 1, RANK = 2};


// Contains the parameters for population evolution. They can be changed at each step.
struct PopulationEvolutionParameters {

	// How many networks are alive at any given time. MUST be a multiple of 
	int nSpecimens;

	// Minimum at 1. If = 1, mutated clone of the parent. No explicit maximum, but bounded by nSpecimens, 
	// and MAX_MATING_DEPTH implicitly. Cost O( n log(n) ).
	int maxNParents;

	// How many consecutive trials a group of Networks experiences, between 2 shuffles of the networks array. 
	int nTrialsPerGroup;

	// How deep the networks are
	int nLayers;

	// layer by layer input sizes of the modules.
	int* inSizes;

	// layer by layer output sizes of the modules.
	int* outSizes;

	// layer by layer number of children per module. Is 0 only at nLayers'th position.
	int* nChildrenPerLayer;

	// layer by layer number of evolved modules. 
	int* nEvolvedModulesPerLayer;

	// layer by layer target fraction of the modules that is replaced at the end of a group stage.
	float* moduleReplacedFractions;

	// target fraction of the networks that is replaced at the end of a group stage.
	float networkReplacedFraction;

	// the fitness at this step, for both network and modules, is an average over the trials of 
	// (1 + voteValue * sign(groupF) * vote) * groupF.
	float voteValue;

	// for all modules and networks, lifetimeFitness(t) = lifetimeFitness(t-1) * accumulatedFitnessDecay + fitness(t)
	// Could be a per layer factor, but there are already enough parameters.
	float accumulatedFitnessDecay;

	//defaults. The fields that are not set here MUST be filled outside !
	PopulationEvolutionParameters() {
		maxNParents = 10;
		nTrialsPerGroup = 5;
		nLayers = 1;
	}
};


struct PhylogeneticNode
{
	// A PhylogeneticNode exists until its depth reaches MAX_MATING_DEPTH. 
	
	// The PNode associated with the module that was "this"'s module's primary parent. nullptr if this
	// is the first generation or the parent was too deep.
	PhylogeneticNode* parent;

	// the position of the associated module in the module list (at its layer). Is -1 if the associated module
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
			d0 > depth )            // the child whose depth increased was not one of the shallowest
		{
			return;
		}
		
		int d1 = depth + 1;
		int minD = MAX_PHYLOGENETIC_DEPTH-1;
		for (int i = 0; i < children.size(); i++) 
		{
			if (minD > children[i]->depth) minD = children[i]->depth;
		}
		depth = minD + 1;

		if (depth == MAX_PHYLOGENETIC_DEPTH) {
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

// A group of a fixed number of individuals, optimized with a genetic algorithm.
class Population {

public:	
	~Population();

	void evolve(int nSteps);


	Population(GroupTrial* trial, PopulationEvolutionParameters& params);


	void setEvolutionParameters(PopulationEvolutionParameters params) {

		if (params.nSpecimens%groupTrial->nAgents != 0) { // To make sure nSpecimens is a multiple of nAgents
			std::cout << "Warning: nSpecimens was not a multiple of nAgents" << std::endl;
			params.nSpecimens = (params.nSpecimens / groupTrial->nAgents) * groupTrial->nAgents;
		}

		this->nSpecimens = params.nSpecimens;
		this->nTrialsPerGroup = params.nTrialsPerGroup;
		this->maxNParents = params.maxNParents;
		this->nLayers = params.nLayers;
		this->inSizes = params.inSizes;
		this->outSizes = params.outSizes;
		this->nChildrenPerLayer = params.nChildrenPerLayer;
		this->nEvolvedModulesPerLayer = params.nEvolvedModulesPerLayer;
		this->moduleReplacedFractions = params.moduleReplacedFractions;
		this->networkReplacedFraction = params.networkReplacedFraction;
		this->voteValue = params.voteValue;
		this->accumulatedFitnessDecay = params.accumulatedFitnessDecay;

		//PhylogeneticNode::maxListSize = params.nParents;
	}

	
	void saveFittestSpecimen();

private:
	GroupTrial* groupTrial;

	// the fitness per group per trial. (1 line = 1 trial)
	float* groupFitnesses;

	// Used to hold the probabilities over networks or modules when creating offsprings. Here to spare
	// allocations. Size max(nEvolvedModulesPerLayer[] & nSpecimens)
	std::unique_ptr<float[]> probabilities;

	// Layer by layer total number of modules in a network. Computed once as a util.
	std::unique_ptr<int[]> nModulesPerNetworkLayer;

#ifdef SPARSE_MUTATION_AND_COMBINATIONS
	// the fraction of the total number of parameters that will be mutated in a module
	// when mutateFloats(mutationsProbabilitiesPerLayer[l]) is called
	std::unique_ptr<float[]> mutationsProbabilitiesPerLayer;
#endif

	// Layer by layer lifetime fitness threshold for modules. Dynamic quantities, adjusted after each
	// replacement step to match the actual replaced fraction to moduleReplenishments more closely at the next step.
	std::unique_ptr<float[]> currentModuleReplacementTreshold;

	float currentNetworkReplacementTreshold;

	// Util for networks.
	int nNodesPerNetwork, inputArraySize, destinationArraySize;

	// To save progress regularly
	int fittestSpecimen;

	// = nSpecimens / trial->nAgentsPerTrial. A util.
	int nGroups;

	Network** networks;

	std::vector<Node_G**> modules;

	std::vector<Node_G*> toBeDestroyedModules;

	//  = PhylogeneticNode[nLayers][nModulesAtThisLayer]
	std::vector<PhylogeneticNode**> phylogeneticTrees;

	// Each group experiences nTrialsPerGroup trials, and the network and module 
	// fitnesses are updated according to the results.
	void evaluateGroups(bool log);


	void replaceNetworks();

	void replaceModules();

	void deleteUnusedModules();

	// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
	Node_G* createChild(PhylogeneticNode* primaryParent, int moduleLayer);
	
	void createPhenotype(Network* n);

	// EVOLUTION PARAMETERS: 
	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	int nSpecimens;
	int nTrialsPerGroup;
	int maxNParents;
	int nLayers;
	int* inSizes;
	int* outSizes;
	int* nChildrenPerLayer;
	int* nEvolvedModulesPerLayer;
	float* moduleReplacedFractions;
	float networkReplacedFraction;
	float voteValue; 
	float accumulatedFitnessDecay;
};