#pragma once

#include "Macros.h"

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


// options still in the files : H init (NodeP), trial uses same seed at reset (in population.cpp, thread loop),
// initial elimination threshold for agents and modules (in system and modulePopulation's constructors )


// Sparse mutations consist in adding a sparse gaussian vector to the network, whose components have
// significant values. Sparse combination replaces each parameter with one of its parents, sampled uniformly
// among those having higher fitness.
#define SPARSE_MUTATION_AND_COMBINATIONS


// étayage élagage dans mes notes (étoffage élagage dans les plus anciennes)
//#define SPRAWL_PRUNE

#ifndef SPRAWL_PRUNE
#define ABCD_ETA // Do not comment this line. one and ony one of SPRAWL_PRUNE and ABCD_ETA must be active 
#endif

// When defined, presynaptic activities of complexNodes (topNode excepted) are a decaying sum of past inputs. This
// sums decays with time and also as a function of the magnitude of the postsynaptic activation.
//#define STDP


//******************* END OF PARAMETERS CHOICES ***************//

// what follows must not be modified, it computes the required memory space for the algorithm.
// (by computing how many parameters each module requires.)



#ifdef STDP
#define STDP_STAT_VECS_01 2 // mu, lambda
//#define STDP_DYN_VECS 1     // accumulated presynacts are not managed like this
#else
#define STDP_STAT_VECS_01 0
#endif

#ifdef ABCD_ETA
#define ABCD_ETA_MATS_R 5 // ABCD + wMod
#define ABCD_ETA_MATS_01 1 // eta
#define ABCD_ETA_MATS_DYNA 2 // H,E
#else
#define ABCD_ETA_MATS_R 0
#define ABCD_ETA_MATS_01 0
#define ABCD_ETA_MATS_DYNA 0
#endif

#ifdef SPRAWL_PRUNE
#define SPRAWL_PRUNE_MATS_R 6     // les deux d (aleph,beth), les deux wMod
#define SPRAWL_PRUNE_MATS_01 2    // les deux etas
#define SPRAWL_PRUNE_MATS_DYNA 4  // H, les deux E, un slot pour d (temporaire)
#else
#define SPRAWL_PRUNE_MATS_R 0
#define SPRAWL_PRUNE_MATS_01 0
#define SPRAWL_PRUNE_MATS_DYNA 0
#endif


#define N_DYNAMIC_MATRICES     (ABCD_ETA_MATS_DYNA + SPRAWL_PRUNE_MATS_DYNA)
#define N_STATIC_MATRICES_01   (ABCD_ETA_MATS_01 + SPRAWL_PRUNE_MATS_01)
#define N_STATIC_MATRICES_R    (ABCD_ETA_MATS_R + SPRAWL_PRUNE_MATS_R)

#define N_DYNAMIC_VECTORS      (0  )
#define N_STATIC_VECTORS_01    (STDP_STAT_VECS_01)
#define N_STATIC_VECTORS_R	   (0  )


