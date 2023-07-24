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

// When defined, CPU functions replace CUDA kernels. Used for debugging purposes.
//#define NO_CUDA

// When defined, networks are evaluated individually (the group trial is ignored, only its innerTrial is used.)
// Used for debugging purposes.
#define NO_GROUP

// When defined, presynaptic activities of complexNodes (topNode excepted) are a decaying sum of past inputs. This
// sums decays with time and also as a function of the magnitude of the postsynaptic activation.
//#define STDP

// mu, lambda
#ifdef STDP
#define STDP_VECS 2
#else
#define STDP_VECS 0
#endif

// Mutations consist in adding a sparse gaussian vector to the network, whose components have
// significant values. Combination replaces each parameter with one of its parents, sampled uniformly
// among those having higher fitness.
#define SPARSE_MUTATION_AND_COMBINATIONS


#define ABC_ETA
#ifdef ABC_ETA
#define N_MATRICES 6
#endif

// kappa
#define N_VECS 1 + STDP_VECS


#define MODULATION_VECTOR_SIZE 1     // DO NOT CHANGE


// Maximum depth of the phylogenetic tree. This means that all pairs of modules used in a combination cannot be
// be further away genetically than MAX_PHYLOGENETIC_DEPTH combinations and mutations. 
// MUST BE >= 1
#define MAX_PHYLOGENETIC_DEPTH 10
