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

	// layer by layer target fraction of the modules that is replaced at the end of a module cycle.
	float* moduleReplacedFractions;

	// target fraction of the networks that is replaced at the end of a network cycle.
	float networkReplacedFraction;

	// the fitness at this step, for both network and modules, is an average over the trials of 
	// (1 + voteValue * sign(groupF) * vote) * groupF.
	float voteValue;

	// for all modules and networks, lifetimeFitness(t) = lifetimeFitness(t-1) * accumulatedFitnessDecay + fitness(t)
	// Could be a per layer factor, but there are already enough parameters.
	float accumulatedFitnessDecay;

	// In [0, 1]. All mutation probabilities are proportional to this value.
	float baseMutationProbability;

	// Positive integer value. Specimens whose phenotypic distance to the primary parent are below it
	// are not used for combination. MUST BE >= 1
	int consanguinityDistance;

	// Whether or not to use an evolved (fixed during lifetime) MLP to pre-process the observations
	// of the environment and post process the dynamic network's decisions.
	bool useInMLP, useOutMLP;

	// >= 1. The number of fully connected layers the MLP has. 
	int inputMLPnLayers, outputMLPnLayers;

	// layer by layer target fraction of the MLPs that is replaced at the end of a module cycle.
	float inMLPReplacedFraction, outMLPReplacedFraction;

	// the layer sizes of the MLPs
	int* inputMLPsizes, *outputMLPsizes;

	// The number of evolved MLPs. 
	int nInMLPs, nOutMLPs;

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
		this->baseMutationProbability = params.baseMutationProbability;
		this->consanguinityDistance = params.consanguinityDistance;

		this->useInMLP = params.useInMLP;
		this->useOutMLP = params.useOutMLP;
		this->inMLPnLayers = params.inputMLPnLayers;
		this->outMLPnLayers = params.outputMLPnLayers;
		this->inMLPReplacedFraction = params.inMLPReplacedFraction;
		this->outMLPReplacedFraction = params.outMLPReplacedFraction; 
		this->inMLPsizes = params.inputMLPsizes;
		this->outMLPsizes = params.outputMLPsizes;
		this->nInMLPs = params.nInMLPs;
		this->nOutMLPs = params.nOutMLPs; 

	}

	
	void saveFittestSpecimen();

private:
	GroupTrial* groupTrial;

	// the fitness per group per trial. (1 line = 1 trial)
	float* groupFitnesses;

	// Pre allocated storage.
	// Used to hold the probabilities over objects when creating offsprings, for
	// either of networks, modules or MLPs. 
	// Size = max(nEvolvedModulesPerLayer[], nSpecimens, nMLPin, nMLPout)
	std::unique_ptr<float[]> probabilities;

	// Layer by layer total number of modules in a network. Computed once as a util.
	std::unique_ptr<int[]> nModulesPerNetworkLayer;

#ifdef SPARSE_MUTATION_AND_COMBINATIONS
	// the fraction of the total number of parameters that will be mutated in a module
	// when mutate(mutationsProbabilitiesPerLayer[l]) is called
	std::unique_ptr<float[]> mutationsProbabilitiesPerLayer;
#endif
	
	float inMLPmutationProbability, outMLPmutationProbability;
	float currentInMLPReplacementTreshold, currentOutMLPReplacementTreshold;
	MLP_G** inMLPs;
	MLP_G** outMLPs;
	std::vector<MLP_G*> toBeDestroyedInMLPs, toBeDestroyedOutMLPs;
	PhylogeneticNode** inMLPphylogeneticTree;
	PhylogeneticNode** outMLPphylogeneticTree;
	void replaceMLPs();
	// inOrOut true if an inMLP must be created, false if it is an outMLP.
	MLP_G* createMLPChild(PhylogeneticNode* primaryParent, bool inOrOut); 
	void deleteUnusedMLPs();
	void zeroMLPsaccumulators();


	// Layer by layer lifetime fitness threshold for modules. Dynamic quantities, adjusted after each
	// replacement step to match the actual replaced fraction to moduleReplenishments more closely at the next step.
	std::unique_ptr<float[]> currentModuleReplacementTreshold;

	float currentNetworkReplacementTreshold;

	// Util for networks.
	int nNodesPerNetwork;

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

	// used only when NO_GROUP is defined. replaces evaluateGroups.
	void evaluateNetsIndividually(bool log);

	void replaceNetworks();

	void replaceModules();

	void deleteUnusedModules();

	void zeroModulesAccumulators();

	// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
	Node_G* createModuleChild(PhylogeneticNode* primaryParent, int moduleLayer);
	
	void createPhenotype(Network* n);


	// CUDA UTILS
	
	// There are groupTrial->nAgents networks, nLayers layers per network, 
	// nModulesPerNetworkLayer[l] nodes per layer, 3 possible destinations 
	// (modulation, children, output) per node, and many matrices per destination. 
	// We want to batch (strided) matrix multiplications, which can be done at the
	// lowest granularity "per layer", i.e. at a given layer, matmuls for  all
	// networks of the group, for all nodes of this layer, are batched. This requires
	// the matrices be contiguous.


	// Vanilla : A,B,C,eta,E,H -> 6 matrix types, so Dim 2 has valid indices [0->5], 
	// and Dim 0 = layer   Dim 1 = destination  
	// The number of matrices may change with different preprocessor directives.
	float**** matrices_LayerDestinationType_CUDA;


	// Same story but for vectors. Stores parameters kappa (STDP_mu, STDP_lambda) 
	// also depends on preprocessor directives.
	float**** vectors_LayerDestinationType_CUDA;



	// Allocates the space for the above matrices and vectors on GPU. To be called once and only once at the 
	// construction of "this".
	void GPUpreallocForNetworks();

	void GPUdeallocForNetworks();






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
	float baseMutationProbability;
	int consanguinityDistance;
	bool useInMLP, useOutMLP; 
	int inMLPnLayers, outMLPnLayers;
	float inMLPReplacedFraction, outMLPReplacedFraction;
	int* inMLPsizes, * outMLPsizes;
	int nInMLPs, nOutMLPs; 
};