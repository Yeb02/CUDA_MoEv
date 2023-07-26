#pragma once
#include "Node_P.h"

Node_P::Node_P(Node_G* _type, Node_G** nodes, int i, int iC, int* nC, int tNC) :
	type(_type),
	toChildren(&_type->toChildren),
	toModulation(&_type->toModulation),
	toOutput(&_type->toOutput)
{
	tNC *= nC[0];
	children.reserve(nC[0]);
	for (int j = 0; j < nC[0]; j++) {
		int child_i = nC[0] * i + j;
		children.emplace_back(nodes[iC + child_i], nodes, child_i, iC + tNC, nC+1, tNC);
	}

	toChildren.randomInitH();
	toModulation.randomInitH();
	toOutput.randomInitH();

	// TotalM is not initialized (i.e. zeroed) because it is set at each inference
	// by either the parent Node_P, or (for topNode) the parent Network.
};


void Node_P::setArrayPointers(float** iA, float** iA_avg, float** dA, float** dA_avg, float** dA_preAvg)
{
	inputArray = *iA;
	inputArray_avg = *iA_avg;
	destinationArray = *dA;
	destinationArray_avg = *dA_avg;


	*iA += type->inputSize + MODULATION_VECTOR_SIZE;
	*iA_avg += type->inputSize + MODULATION_VECTOR_SIZE;

	*dA += type->outputSize + MODULATION_VECTOR_SIZE;
	*dA_avg += type->outputSize + MODULATION_VECTOR_SIZE;


#ifdef STDP
	destinationArray_preAvg = *dA_preAvg;
	*dA_preAvg += type->outputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < children.size(); i++) {
		
		*iA += children[i].type->outputSize;
		*iA_avg += children[i].type->outputSize;

		*dA += children[i].type->outputSize;
		*dA_avg += children[i].type->outputSize;
#ifdef STDP
		*dA_preAvg += children[i].type->outputSize;
#endif

	}

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(iA, iA_avg, dA, dA_avg, dA_preAvg);
	}
}


void Node_P::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		children[i].preTrialReset();
	}

	toChildren.zeroE();
	toModulation.zeroE();
	toOutput.zeroE();

	// TotalM is not reinitialized (i.e. zeroed) because it is set at each inference
	// by either the parent Node_P, or (for topNode) the parent Network.
}



void Node_P::forward(bool firstCall) {
	
	// Modulation could be done on CPU while the GPU handles the bulk of the work. TODO benchmark to
	// see if it is worth the efforrt (and 1 cycle of delay in modulation !!)

	// This lambda is used for all 3 propagations : modulation, children, output.
	// It should be a GPU kernel.
	auto forwardAndLocalUpdates = [this, firstCall](InternalConnexion_P& icp, int offset)
	{
		// Variables defined for readability.

		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;

		// destinationArray instead of inputArray yields better results ????? TODO
		// Buffer overread
		float* modulation = inputArray + type->inputSize; 

		float* H = icp.H.get();
		float* E = icp.E.get();

		
		float* A = icp.type->A.get();
		float* B = icp.type->B.get();
		float* C = icp.type->C.get();
		float* eta = icp.type->eta.get();

		float* D = icp.type->D.get();
		float* F = icp.type->F.get();
		float* G = icp.type->G.get();

		float* kappa = icp.type->kappa.get();
#ifdef STDP
		float* mu = icp.type->STDP_mu.get();
		float* lambda = icp.type->STDP_lambda.get();
#endif
		
		float* dA = destinationArray + offset;
		float* dA_avg = destinationArray_avg + offset;
#ifdef STDP
		float* dA_preAvg = destinationArray_preAvg + offset;
#endif
		
		// All those steps could be individually vectorized. In one block for compacity, as the CPU version 
		// is not designed to achieve high performances. When explicitly mentionned, 2 steps can be swapped. 
		// Otherwise, order matters.

		for (int i = 0; i < nl; i++) {
			int lineOffset = i * nc;

			// 6 and 7, testing in front but can be in the back. If in the back, the 
			// if (firstCall) is no longer necessary. But recommended ?
			if (!firstCall && true) [[likely]]
			{
				for (int j = 0; j < nc; j++) {
					int matID = lineOffset + j;

					// 6:  
					// inputArray[j] - inputArray_avg[j] does not depend on the line and can therefore be precomputed 
					// (here each substraction is computed redundantly nLines times.)
					E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
						(A[matID] * inputArray[j] * dA[i] + B[matID] * inputArray[j] + C[matID] * dA[i]  
							+ 5.0f * (inputArray[j] - inputArray_avg[j]) * 
							(dA[i] + D[matID]* dA_avg[i]) * 
							(F[matID] + H[matID] * G[matID])
						);
						


					// 7:
					H[matID] += E[matID] * modulation[0];
					H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
				}
			}


			// 0:
			float preSynAct = 0.0f; 
			for (int j = 0; j < nc; j++) {
				preSynAct += H[lineOffset+j] * inputArray[j];
			}

			// 1:
#ifdef STDP
			if (firstCall) [[unlikely]] {
				dA_preAvg[i] = preSynAct;
			}
			else [[likely]] {
				dA_preAvg[i] = dA_preAvg[i] * (1.0f - mu[i]) + preSynAct; // * mu[i] ? TODO
			}
#endif

			// 2:
			dA_avg[i] = dA_avg[i] * (1.0f - kappa[i]) + dA[i] * kappa[i];

			// 3:
#ifdef STDP
			dA[i] = tanhf(dA_preAvg[i]);
#else
			dA[i] = tanhf(preSynAct);
#endif

			// 4:
			if (firstCall) [[unlikely]] {
				dA_avg[i] = dA[i];
			}

			// 5:
#ifdef STDP
			// acc_src's magnitude decreases when there is a significant activation.
			// * 4.0f because lambda is in [0,1] so the magnitude would be too limited. 
			dA_preAvg[i] -= lambda[i] * powf(dA[i], 2.0f * 1.0f + 1.0f) * 4.0f;
#endif

			if (false) 
			{
				for (int j = 0; j < nc; j++) {
					int matID = lineOffset + j;

					// 6:  
					// inputArray[j] - inputArray_avg[j] does not depend on the line and can therefore be precomputed 
					// (here each substraction is computed redundantly nLines times.)
					E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
						(A[matID] * inputArray[j] * dA[i] + B[matID] * inputArray[j] + C[matID] * dA[i]
							+ 5.0f * (inputArray[j] - inputArray_avg[j]) *
							(dA[i] + D[matID] * dA_avg[i]) *
							(F[matID] + H[matID] * G[matID])
						);


					// 7:
					H[matID] += E[matID] * modulation[0];
					H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
				}
			}
		}
	};


	// MODULATION A
	{
		forwardAndLocalUpdates(toModulation, type->outputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			inputArray[i + type->inputSize] = destinationArray[i + type->outputSize];
		}
	}


	// CHILDREN
	if (children.size() != 0) {

		forwardAndLocalUpdates(toChildren, type->outputSize + MODULATION_VECTOR_SIZE);

		// Each child must receive its modulation, input, and accumulated input. 
		int id = type->outputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < children.size(); i++) {

			std::copy(
				destinationArray + id,
				destinationArray + id + children[i].type->inputSize,
				children[i].inputArray
			);

			std::copy(
				destinationArray_avg + id,
				destinationArray_avg + id + children[i].type->inputSize,
				children[i].inputArray_avg
			);
			

			id += children[i].type->inputSize;
		}

		
		for (int i = 0; i < children.size(); i++) {
			children[i].forward(firstCall);
		}

		// children's output and average output must be retrieved.
		id = type->inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < children.size(); i++) {
			std::copy(
				inputArray + id,
				inputArray + id + children[i].type->outputSize,
				children[i].destinationArray
			);

			std::copy(
				inputArray_avg + id,
				inputArray_avg + id + children[i].type->outputSize,
				children[i].destinationArray_avg
			);

			id += children[i].type->outputSize;
		}
	}


	// MODULATION B
	if (children.size() != 0) {
		forwardAndLocalUpdates(toModulation, type->outputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			inputArray[i + type->inputSize] = destinationArray[i + type->outputSize];
		}
	}

	
	// OUTPUT
	forwardAndLocalUpdates(toOutput, 0);
}
