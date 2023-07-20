#pragma once


////////////////////////////////////
///// USER COMPILATION CHOICES /////
////////////////////////////////////


// Comment/uncomment or change the value of various preprocessor directives
// to compile different versions of the code. Or use the -D flag.


// Define the trials on which to evolve. One and only one must be defined if compiling the .exe (or tweak main()). 
// These do not affect the DLL.
#define CARTPOLE_T
//#define XOR_T
//#define TMAZE_T
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T
//#define ROCKET_SIM_T 


// Mutations consist in adding a sparse gaussian vector to the network, whose components have
// significant values. Combination replaces each parameter with one of its parents, sampled uniformly
// among those having higher fitness.
#define SPARSE_MUTATION_AND_COMBINATIONS


#define BASE_MUTATION_P .2f


#define MODULATION_VECTOR_SIZE 1     // DO NOT CHANGE


// Maximum number of generations since last common ancestor of two  (of the) specimens combined to form a new specimen. >= 2.
#define MAX_MATING_DEPTH 10


// Positive integer value. Specimens whose phenotypic distance to the primary parent are below it are not used for combination.
#define CONSANGUINITY_DISTANCE 0


// parameters that have values in the range [0,1] are initialized with mean DECAY_PARAMETERS_BIAS
// These parameters (denoted µ) are typically used in exponential moving average updates, i.e.
// X(t+1) = X(t) * (1-µ)  +  µ * ....
#define DECAY_PARAMETERS_INIT_BIAS .2f


// When defined, presynaptic activities of complexNodes (topNode excepted) are an exponential moving average. Each node 
// be it Modulation, complex, memory or output has an evolved parameter (STDP_decay) that parametrizes the average.
// WARNING only compatible with N_ACTIVATIONS = 1, I havent implemented all the derivatives in complexNode_P::forward yet
#define STDP

