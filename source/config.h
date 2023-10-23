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
//#define TEACHING_T
//#define TMAZE_T
//#define N_LINKS_PENDULUM_T
//#define MEMORY_T
//#define ROCKET_SIM_T 


// options still in the files : 
// -trial uses same seed at reset (in system.cpp, thread loop),



// SPARSE_MUTATIONS and SPARSE_COMBINATIONS should be both enabled, or both disabled, for best results.
// WARNING As of now, keep them defined, as their implementations are incomplete.

// Sparse mutations consist in adding a sparse gaussian vector to the module's parameters, whose components have
// "high" values. The alternative is to add a dense gaussian vector, of similar magnitude.
#define SPARSE_MUTATIONS

// Sparse combination replaces each parameter in a module individually with one of its parents, sampled proportionally
// to their relative fitness. The alternative is a linear combination of the parents parameters, whose coefficients are
// the same for each parameter (and functions of the relative fitness. Sum to 1.).
#define SPARSE_COMBINATIONS


#define PREDICTIVE_CODING
#ifdef PREDICTIVE_CODING

// Instead of a fixed, global variance equal to 1 across the activations, modules and agents, 
// each sigma is now evolved and per activation. (But also dynamically updated, as thetas and biases.)
// This can be interpred as using a per activation variance not fixed to 1, but without correlation 
// between activations on the same layer. It is used to multiply epsilons before any further computations.
// The vector contains the inverses to speed up inference.
// As in Friston (2005), invSigma takes values in [0,1], i.e. sigma >= 1.
// TODO insert modulation ----------------------------------------------------------------------------- TODO
#define ACTIVATION_VARIANCE


// Still looking for a satisfactory solution. 
//#define MODULATED

// Places the observation at layer 0, and the action at layer L. (normally the other way around)
// A network that is not reversed, not modulated, and whose root module does not have children, is effectively a fixed-weight one.
//#define ACTION_L_OBS_O




#else
// étayage élagage dans mes notes (étoffage élagage dans les plus anciennes)
//#define SPRAWL_PRUNE

#ifndef SPRAWL_PRUNE
#define ABCD_ETA // Do not comment this line. one and ony one of SPRAWL_PRUNE and ABCD_ETA must be active 
#endif

// When defined, presynaptic activities of complexNodes (topNode excepted) are a decaying sum of past inputs. This
// sums decays with time and also as a function of the magnitude of the postsynaptic activation.
//#define STDP


// younger networks and agents have a slight chance of escaping slaughter.
//#define YOUNG_AGE_BONUS
#endif


//******************* END OF PARAMETERS CHOICES ***************//

// what follows must not be modified, it computes the required memory space for the algorithm
// (by computing how many parameters each module requires) and other logic


#ifdef PREDICTIVE_CODING
#define MODULE_P PC_Node_P
#define MODULE PC_Node_G
#define MODULE_PARAMETERS PC_Node_GFixedParameters
#define AGENT PC_Network
#define AGENT_PARAMETERS PC_NetworkParameters
#else
#define MODULE_P HebbianNode_P
#define MODULE HebbianNode_G
#define MODULE_PARAMETERS HebbianNode_GFixedParameters
#define AGENT HebbianNetwork
#define AGENT_PARAMETERS HebbianNetworkParameters
#endif

#ifdef ACTIVATION_VARIANCE
#define SIGMA_VEC 1
#else
#define SIGMA_VEC 0
#endif

#ifdef MODULATED
#define MOD_MAT 1
#else
#define MOD_MAT 0
#endif

#ifdef PREDICTIVE_CODING // TODO if you change the number of matrices you must also change the initialization of InternalConnexion_P
#define PC_MATS_R (1 + MOD_MAT) // weights, modulation?
#define PC_VECS_R 1  // bias
#define PC_VECS_01 SIGMA_VEC // variance?
#define PC_MATS_DYNA 1 // weights
#define PC_VECS_DYNA (1 + SIGMA_VEC) // bias, variance?
#else
#define PC_MATS_R 0
#define PC_VECS_R 0
#define PC_MATS_DYNA 0
#define PC_VECS_DYNA 0
#endif


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




#define N_DYNAMIC_MATRICES     (ABCD_ETA_MATS_DYNA + SPRAWL_PRUNE_MATS_DYNA + PC_MATS_DYNA)
#define N_STATIC_MATRICES_01   (ABCD_ETA_MATS_01 + SPRAWL_PRUNE_MATS_01)
#define N_STATIC_MATRICES_R    (ABCD_ETA_MATS_R + SPRAWL_PRUNE_MATS_R + PC_MATS_R)

#define N_DYNAMIC_VECTORS      (PC_VECS_DYNA)
#define N_STATIC_VECTORS_01    (STDP_STAT_VECS_01 + PC_VECS_01)
#define N_STATIC_VECTORS_R	   (PC_VECS_R)


