#pragma once

#include "MoEvCore.h"
#ifndef PREDICTIVE_CODING

#include "HebbianNode_P.h"


HebbianNode_P::HebbianNode_P(HebbianNode_G* _type, HebbianNode_G** nodes, int i, int iC, int* nC, int tNC) :
	type(_type),
	toChildren(&_type->toChildren),
	toOutput(&_type->toOutput),
	childrenInputV(nullptr, 0), concInputV(nullptr, 0), outputV(nullptr, 0)
{
	tNC *= nC[0];
	children.reserve(nC[0]);
	for (int j = 0; j < nC[0]; j++) {
		int child_i = nC[0] * i + j;
		children.emplace_back(nodes[iC + child_i], nodes, child_i, iC + tNC, nC+1, tNC);
	}
};


void HebbianNode_P::setArrayPointers(float** iA, float** dArr, float** dArr_preSynAvg)
{

	// placement new is the recommended way: https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html
	new (&childrenInputV) MVector(*dArr + type->outputSize, toChildren.type->nRows);
	new (&concInputV) MVector(*iA, toOutput.type->nColumns); // (toOutput and toChildren have the same number of columns)
	new (&outputV) MVector(*dArr, toOutput.type->nRows);

	*iA += type->inputSize;
	*dArr += type->outputSize;

#ifdef STDP
	outCinActivations_preSynAvg = *dArr_preSynAvg;
	*dArr_preSynAvg += type->outputSize;
#endif

	for (int i = 0; i < children.size(); i++) {
		
		*iA += children[i].type->inputSize;

		*dArr += children[i].type->outputSize;
#ifdef STDP
		*dArr_preSynAvg += children[i].type->outputSize;
#endif

	}

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(iA, dArr, dArr_preSynAvg);
	}
}


void HebbianNode_P::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		children[i].preTrialReset();
	}

	toChildren.preTrialReset();
	toOutput.preTrialReset();

}


#ifdef ABCD_ETA
void HebbianNode_P::forward() {


	auto forwardAndLocalUpdates = [this](InternalConnexion_P& icp, MVector& dstV)
	{

		constexpr bool HUpdateFirst = true;
		if (!HUpdateFirst)
		{
			dstV = (icp.matrices[0] * concInputV).array().tanh();
		}

		icp.modulationV.noalias() = icp.type->matricesR[4] * concInputV;

		icp.matrices[1] = (1.0f - icp.type->matrices01[0].array()) * icp.matrices[1].array();

		/*std::cerr << icp.matrices[1].rows() << icp.matrices[1].cols() << std::endl;
		std::cerr << icp.type->matrices01[0].rows() << icp.type->matrices01[0].cols() << std::endl;
		std::cerr << icp.type->matricesR[0].rows() << icp.type->matricesR[0].cols() << std::endl;
		std::cerr << icp.type->matricesR[1].rows() << icp.type->matricesR[1].cols() << std::endl;
		std::cerr << icp.type->matricesR[2].rows() << icp.type->matricesR[2].cols() << std::endl;
		std::cerr << icp.type->matricesR[3].rows() << icp.type->matricesR[3].cols() << std::endl;
		std::cerr << dstV.rows() << dstV.cols() << std::endl;
		std::cerr << concInputV.rows() << concInputV.cols() << std::endl;
		Eigen::MatrixXf aaa = icp.type->matricesR[2].array().rowwise() * concInputV.array().transpose();
		std::cerr << aaa.rows() << aaa.cols() << std::endl;
		Eigen::MatrixXf bbb = icp.type->matricesR[1].array().colwise() * dstV.array();
		std::cerr << bbb.rows() << bbb.cols() << std::endl;*/


		icp.matrices[1].noalias() += (icp.type->matrices01[0].array() * (
			(dstV * concInputV.transpose()).array() * icp.type->matricesR[0].array() +
			icp.type->matricesR[1].array().rowwise() * concInputV.array().transpose() +  
			icp.type->matricesR[2].array().colwise() * dstV.array() +	      
			icp.type->matricesR[3].array())).matrix();						

		icp.matrices[0].noalias() += (icp.matrices[1].array().colwise() * icp.modulationV.array()).matrix();
		icp.matrices[0] = icp.matrices[0].cwiseMin(4.0f).cwiseMax(-4.0f);
		
		if (HUpdateFirst)
		{
			dstV = (icp.matrices[0] * concInputV).array().tanh();
		}
	};


	// INPUT_(CHILDREN'S OUTPUT)   TO    (CHILDREN'S INPUT)
	// THEN CHILDREN'S FORWARD
	if (children.size() != 0) {

		forwardAndLocalUpdates(toChildren, childrenInputV);

		// Each child must be sent its input and accumulated input. 
		float* apt = childrenInputV.data();
		for (int i = 0; i < children.size(); i++) {

			std::copy(
				apt,
				apt + children[i].type->inputSize,
				children[i].concInputV.data()
			);

			apt += children[i].type->inputSize;
		}


		for (int i = 0; i < children.size(); i++) {
			children[i].forward();
		}

		// children's output and average output must be retrieved.
		apt = concInputV.data() + type->inputSize;
		for (int i = 0; i < children.size(); i++) {
			std::copy(
				apt,
				apt + children[i].type->outputSize,
				children[i].outputV.data()
			);

			apt += children[i].type->outputSize;
		}
	}

	// INPUT_(CHILDREN'S OUTPUT)   TO    OUTPUT
	forwardAndLocalUpdates(toOutput, outputV);
}
#endif

// better safe than sorry.
#ifdef PRE_EIGEN_CHANGES

#ifdef STDP
		float* mu = icp.type->vectors01[0];
		float* lambda = icp.type->vectors01[1];
#endif
		
		float* dArr = outCinActivations + offset;
#ifdef STDP
		float* dArr_preSynAvg = outCinActivations_preSynAvg + offset;
#endif
		
			float preSynAct = 0.0f; 
			for (int j = 0; j < nc; j++) {
				preSynAct += H[lineOffset+j] * inCoutActivations[j];
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
		}
	};

void HebbianNode_P::forward()
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

		float* dArr = outCinActivations + offset;
#ifdef STDP
		float* dArr_preSynAvg = outCinActivations_preSynAvg + offset;
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
				modulation += wModA[lineOffset + j] * inCoutActivations[j];
			}

			// 3
			for (int j = 0; j < nc; j++) {
				matID = lineOffset + j;

				EA[matID] = (1.0f - etaA[matID]) * EA[matID] + etaA[matID] * inCoutActivations[j] * dArr[i] *
					(dBuffer[matID] * SinvY2 - perRowBuffer[i] * perColumnBuffer[j]);


				H[matID] += EA[matID] * modulation;
				H[matID] = std::clamp(H[matID], -4.0f, 4.0f);
			}



			// 4
			float preSynAct = 0.0f;
			for (int j = 0; j < nc; j++) {
				preSynAct += H[lineOffset + j] * inCoutActivations[j];
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
			perColumnBuffer[j] = inCoutActivations[j] * inCoutActivations[j];
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
				EB[matID] = (1.0f - etaB[matID]) * EB[matID] + etaB[matID] * inCoutActivations[j] * dArr[i] *
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
				outCinActivations + id,
				outCinActivations + id + children[i].type->inputSize,
				children[i].inCoutActivations
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
				inCoutActivations + id,
				inCoutActivations + id + children[i].type->outputSize,
				children[i].outCinActivations
			);

			id += children[i].type->outputSize;
		}
	}

	// OUTPUT
	forwardAndLocalUpdates(toOutput, 0);
}
#endif

#endif
