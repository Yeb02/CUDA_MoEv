#pragma once
#include "Node_P.h"

Node_P::Node_P(Node_G* _type, Node_G** nodes, int i, int iC, int* nC, int tNC) :
	type(_type),
	toChildren(&_type->toChildren),
	toOutput(&_type->toOutput)
{
	tNC *= nC[0];
	children.reserve(nC[0]);
	for (int j = 0; j < nC[0]; j++) {
		int child_i = nC[0] * i + j;
		children.emplace_back(nodes[iC + child_i], nodes, child_i, iC + tNC, nC+1, tNC);
	}


	// TotalM is not initialized (i.e. zeroed) because it is set at each inference
	// by either the parent Node_P, or (for topNode) the parent Network.
};


void Node_P::setArrayPointers(float** iA, float** dArr, float** dArr_preSynAvg)
{
	inputArray = *iA;
	destinationArray = *dArr;

	*iA += type->inputSize;
	*dArr += type->outputSize;

#ifdef STDP
	destinationArray_preSynAvg = *dArr_preSynAvg;
	*dArr_preSynAvg += type->outputSize;
#endif

	for (int i = 0; i < children.size(); i++) {
		
		*iA += children[i].type->outputSize;

		*dArr += children[i].type->outputSize;
#ifdef STDP
		*dArr_preSynAvg += children[i].type->outputSize;
#endif

	}

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(iA, dArr, dArr_preSynAvg);
	}
}


void Node_P::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		children[i].preTrialReset();
	}

	toChildren.zeroE();
	toOutput.zeroE();

	// TotalM is not reinitialized (i.e. zeroed) because it is set at each inference
	// by either the parent Node_P, or (for topNode) the parent Network.
}



#ifdef ABCD_ETA
void Node_P::forward() {


	auto forwardAndLocalUpdates = [this](InternalConnexion_P& icp, int offset)
	{
		// Variables defined for readability.

		int nr = icp.type->nRows;
		int nc = icp.type->nColumns;

		float* H = icp.matrices[0];
		float* E = icp.matrices[1];

		
		float* A = icp.type->matricesR[0];
		float* B = icp.type->matricesR[1];
		float* C = icp.type->matricesR[2];
		float* D = icp.type->matricesR[3];
		float* wMod = icp.type->matricesR[4];

		float* eta = icp.type->matrices01[0];

#ifdef STDP
		float* mu = icp.type->vectors01[0];
		float* lambda = icp.type->vectors01[1];
#endif
		
		float* dArr = destinationArray + offset;
#ifdef STDP
		float* dArr_preSynAvg = destinationArray_preSynAvg + offset;
#endif
		
		// All those steps could be individually vectorized. In one block for compacity, as the CPU version 
		// is not designed to achieve high performances. When explicitly mentionned, 2 steps can be swapped. 
		// Otherwise, order matters.

		for (int i = 0; i < nr; i++) {
			int lineOffset = i * nc;

			// 6 and 7, testing in front but can be in the back.
			if (true) 
			{
				float modulation = 0.0f;
				for (int j = 0; j < nc; j++) {
					modulation += wMod[lineOffset + j] * inputArray[j];
				}

				for (int j = 0; j < nc; j++) {
					int matID = lineOffset + j;

					// 6:  
					E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
						(A[matID] * inputArray[j] * dArr[i] + B[matID] * inputArray[j] + C[matID] * dArr[i] + D[matID]);


					// 7:
					H[matID] += E[matID] * modulation;
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
			
			dArr_preSynAvg[i] = dArr_preSynAvg[i] * (1.0f - mu[i]) + preSynAct; // * mu[i] ? TODO
			
#endif

			// 3:
#ifdef STDP
			dArr[i] = tanhf(dArr_preSynAvg[i]);
#else
			dArr[i] = tanhf(preSynAct);
#endif

			
			// 5:
#ifdef STDP
			// acc_src's magnitude decreases when there is a significant activation.
			// * 4.0f because lambda is in [0,1] so the magnitude would be too limited. 
			dArr_preSynAvg[i] -= lambda[i] * powf(dArr[i], 2.0f * 1.0f + 1.0f) * 4.0f;
#endif

			if (false)
			{
				float modulation = 0.0f;
				for (int j = 0; j < nc; j++) {
					modulation += wMod[lineOffset + j] * inputArray[j];
				}

				for (int j = 0; j < nc; j++) {
					int matID = lineOffset + j;

					// 6:  
					E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
						(A[matID] * inputArray[j] * dArr[i] + B[matID] * inputArray[j] + C[matID] * dArr[i] + D[matID]);


					// 7:
					H[matID] += E[matID] * modulation;
					H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
				}
			}
		}
	};


	// CHILDREN
	if (children.size() != 0) {

		forwardAndLocalUpdates(toChildren, type->outputSize);

		// Each child must receive its modulation, input, and accumulated input. 
		int id = type->outputSize;
		for (int i = 0; i < children.size(); i++) {

			std::copy(
				destinationArray + id,
				destinationArray + id + children[i].type->inputSize,
				children[i].inputArray
			);

			id += children[i].type->inputSize;
		}

		
		for (int i = 0; i < children.size(); i++) {
			children[i].forward();
		}

		// children's output and average output must be retrieved.
		id = type->inputSize;
		for (int i = 0; i < children.size(); i++) {
			std::copy(
				inputArray + id,
				inputArray + id + children[i].type->outputSize,
				children[i].destinationArray
			);

			id += children[i].type->outputSize;
		}
	}
	
	// OUTPUT
	forwardAndLocalUpdates(toOutput, 0);
}
#elif defined(SPRAWL_PRUNE) 
void Node_P::forward() 
{
	constexpr float epsilon = .1f;

	auto forwardAndLocalUpdates = [this](InternalConnexion_P& icp, int offset)
	{
		constexpr float epsilon = .1f;
		
		// Variables defined for readability.

		int nr = icp.type->nRows;
		int nc = icp.type->nColumns;

		float* H = icp.matrices[0];
		float* EA = icp.matrices[1];
		float* EB = icp.matrices[2];

		float* dBuffer = icp.matrices[3];

		float* perRowBuffer = icp.tempBuffer1.get();
		float* perColumnBuffer = icp.tempBuffer2.get();

		float* alephA = icp.type->matricesR[0];
		float* alephB = icp.type->matricesR[1];
		float* bethA = icp.type->matricesR[2];
		float* bethB = icp.type->matricesR[3];
		float* wModA = icp.type->matricesR[4];
		float* wModB = icp.type->matricesR[5];

		float* etaA = icp.type->matrices01[0];
		float* etaB = icp.type->matrices01[1];

#ifdef STDP
		float* mu = icp.type->vectors01[0];
		float* lambda = icp.type->vectors01[1];
#endif

		float* dArr = destinationArray + offset;
#ifdef STDP
		float* dArr_preSynAvg = destinationArray_preSynAvg + offset;
#endif

		// All those steps could be individually vectorized. In one block for compacity, as the CPU version 
		// is not designed to achieve high performances. When explicitly mentionned, 2 steps can be swapped. 
		// Otherwise, order matters.
		
		// 0 
		int matID = 0;
		float SinvY2 = 0.0f;
		for (int i = 0; i < nr; i++) {
			float y2 = dArr[i] * dArr[i];
			for (int j = 0; j < nc; j++) {
				// since dij is multiplied by yi*yi everywhere it appears in the formula, 
				// I pre multiply in place each line of the d matrix by y².
				dBuffer[matID] = (H[matID] * alephA[matID] + bethA[matID]) * y2; 
			}
			perRowBuffer[i] = 1.0f / (y2 + epsilon); 
			SinvY2 += perRowBuffer[i];
		}

		// 1
		for (int j = 0; j < nc; j++) {
			float sdy2 = 0.0f;
			for (int i = 0; i < nr; i++) {
				sdy2 += dBuffer[j + i * nc];
			}
			perColumnBuffer[j] = sdy2;
		}
		
		
		for (int i = 0; i < nr; i++) {
			int lineOffset = i * nc;

			// 2
			float modulation = 0.0f;
			for (int j = 0; j < nc; j++) {
				modulation += wModA[lineOffset + j] * inputArray[j];
			}

			// 3
			for (int j = 0; j < nc; j++) {
				matID = lineOffset + j;

				EA[matID] = (1.0f - etaA[matID]) * EA[matID] + etaA[matID] * inputArray[j] * dArr[i] *
					(dBuffer[matID] * SinvY2 - perRowBuffer[i] * perColumnBuffer[j]);


				H[matID] += EA[matID] * modulation;
				H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
			}



			// 4
			float preSynAct = 0.0f;
			for (int j = 0; j < nc; j++) {
				preSynAct += H[lineOffset + j] * inputArray[j];
			}

			// 5
#ifdef STDP
			dArr_preSynAvg[i] = dArr_preSynAvg[i] * (1.0f - mu[i]) + preSynAct; // * mu[i] ? TODO
			dArr[i] = tanhf(dArr_preSynAvg[i]);
#else
			dArr[i] = tanhf(preSynAct);
#endif


			// 6
#ifdef STDP
			// acc_src's magnitude decreases when there is a significant activation.
			// * 4.0f because lambda is in [0,1] so the magnitude would be too limited. 
			// tanh(x)^3 = tanh(x)*(1-tanh'(x)), 1 = max(tanh'), tanh'>0.
			dArr_preSynAvg[i] -= lambda[i] * powf(dArr[i], 2.0f * 1.0f + 1.0f) * 4.0f;
#endif
		}

		// 7
		for (int j = 0; j < nc; j++) {
			perColumnBuffer[j] = inputArray[j] * inputArray[j];
		}
		matID = 0;
		for (int i = 0; i < nr; i++) {
			float sdx2 = 0.0f;
			float sHx2 = 0.0f;
			for (int j = 0; j < nc; j++) {
				dBuffer[matID] = H[matID] * alephB[matID] + bethB[matID];
				sdx2 += dBuffer[matID] * perColumnBuffer[j];
				sHx2 += H[matID] * perColumnBuffer[j];
				matID++;
			}
			perRowBuffer[i] = sdx2;
			matID -= nc;
			for (int j = 0; j < nc; j++) {
				dBuffer[matID] *= sHx2;
				matID++;
			}
		}

		// 8
		matID = 0;
		for (int j = 0; j < nc; j++) {
			float modulation = 0.0f;
			for (int i = 0; i < nr; i++) {
				// it is not specified anywhere that wModB has the same dimensions as the other matrices,
				// beside maybe in the amplitudes at initialisation and mutation. We ignore it and treat it
				// as if it had transposed dimensions, i.e. nCols rows and nRows columns. This way we can do
				// a matrix*vector multiplication, instead of a vector*matrix multiplication.
				modulation += wModB[matID] * dArr[i]; 
				matID++;
			}
			perColumnBuffer[j] = modulation;
		}

		// 9
		matID = 0;
		for (int i = 0; i < nr; i++) {
			for (int j = 0; j < nc; j++) {
				EB[matID] = (1.0f - etaB[matID]) * EB[matID] + etaB[matID] * inputArray[j] * dArr[i] *
					(dBuffer[matID] - H[matID] * perRowBuffer[i]);


				H[matID] += EB[matID] * perColumnBuffer[j];
				H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
				matID++;
			}
		}


	};


	// CHILDREN
	if (children.size() != 0) {

		forwardAndLocalUpdates(toChildren, type->outputSize);

		// Each child must receive its modulation, input, and accumulated input. 
		int id = type->outputSize;
		for (int i = 0; i < children.size(); i++) {

			std::copy(
				destinationArray + id,
				destinationArray + id + children[i].type->inputSize,
				children[i].inputArray
			);

			id += children[i].type->inputSize;
		}


		for (int i = 0; i < children.size(); i++) {
			children[i].forward();
		}

		// children's output and average output must be retrieved.
		id = type->inputSize;
		for (int i = 0; i < children.size(); i++) {
			std::copy(
				inputArray + id,
				inputArray + id + children[i].type->outputSize,
				children[i].destinationArray
			);

			id += children[i].type->outputSize;
		}
	}

	// OUTPUT
	forwardAndLocalUpdates(toOutput, 0);
}
#elif defined(PREDICTIVE_CODING)
void Node_P::forward(bool firstCall)
{

}
#endif

