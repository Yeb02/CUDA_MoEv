#include "Node_P.h"



Node_P::Node_P(Node_G* _type) : 
	type(_type),
	toChildren(&_type->toChildren),
	toModulation(&_type->toModulation),
	toOutput(&_type->toOutput)
{
	children.reserve(type->children.size());
	for (int i = 0; i < type->children.size(); i++) {
		children.emplace_back(type->children[i]);
	}

	toChildren.randomInitH();
	toModulation.randomInitH();
	toOutput.randomInitH();

	// TotalM is not initialized (i.e. zeroed) here because a call to preTrialReset() 
	// must be made before any forward pass. 
};


void Node_P::setArrayPointers(float** post_syn_acts, float** pre_syn_acts, float** aa, float** acc_pre_syn_acts) {

	// TODO ? if the program runs out of heap memory, one could make it so that a node does not store its own 
	// output. But prevents in place matmul, and complexifies things.

	postSynActs = *post_syn_acts;
	preSynActs = *pre_syn_acts;


	*post_syn_acts += type->inputSize + MODULATION_VECTOR_SIZE;
	*pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;


#ifdef SATURATION_PENALIZING
	averageActivation = *aa;
	*aa += MODULATION_VECTOR_SIZE;
#endif

#ifdef STDP
	accumulatedPreSynActs = *acc_pre_syn_acts;
	*acc_pre_syn_acts += type->outputSize + MODULATION_VECTOR_SIZE;
#endif

	for (int i = 0; i < children.size(); i++) {
		*post_syn_acts += children[i].type->outputSize;
		*pre_syn_acts += children[i].type->inputSize;
#ifdef STDP
		*acc_pre_syn_acts += children[i].type->inputSize;
#endif
#ifdef SATURATION_PENALIZING
		*aa += children[i].type->inputSize;
#endif
	}
	for (int i = 0; i < memoryChildren.size(); i++) {
		memoryChildren[i].setArrayPointers(*post_syn_acts, *pre_syn_acts, totalM, aa, acc_pre_syn_acts);
		*post_syn_acts += memoryChildren[i].type->outputSize;
		*pre_syn_acts += memoryChildren[i].type->inputSize;
	}

	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(post_syn_acts, pre_syn_acts, aa, acc_pre_syn_acts);
	}
}


void Node_P::preTrialReset() {

	for (int i = 0; i < children.size(); i++) {
		children[i].preTrialReset();
	}

	toChildren.zeroE();
	toModulation.zeroE();
	toOutput.zeroE();
}


#ifdef SATURATION_PENALIZING
void Node_P::setglobalSaturationAccumulator(float* globalSaturationAccumulator) {
	this->globalSaturationAccumulator = globalSaturationAccumulator;
	for (int i = 0; i < children.size(); i++) {
		children[i].setglobalSaturationAccumulator(globalSaturationAccumulator);
	}
}
#endif


void Node_P::forward(bool firstCall) {
	
	// TODO there are no reasons not to propagate several times through MODULATION and MEMORY, for instance:
	// MODULATION -> MEMORY -> COMPLEX -> MODULATION -> MEMORY -> OUTPUT ...
	// And it can even be node specific. To be evolved ?

#ifdef SATURATION_PENALIZING
	constexpr float saturationExponent = 6.0f; 
#endif

#ifdef DROPOUT
	toChildren.dropout();
	toMemory.dropout();
	toOutput.dropout();
	toModulation.dropout();
#endif

	// STEP 1 to 6: for each type of node in the sequence:
	// 1_Modulation -> 2_Memory -> 3_Complex -> 4_Modulation -> 5_Memory -> 6_output
	
	// - propagate postSynActs in preSynActs of the corresponding node(s), and apply their non linearities.
	// - apply hebbian update to the involved connexion matrices
	// - apply all nodes of the type's forward
	
	// This could be done simultaneously for all types, but doing it this way drastically speeds up information transmission
	// through the network. 


	// These 3 lambdas, hopefully inline, avoid repetition, as they are used for each child type.

	auto propagate = [this](InternalConnexion_P& icp, float* destinationArray)
	{
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* H = icp.H.get();
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();

#ifdef RANDOM_WB
		float* w = icp.w.get();
		float* b = icp.biases.get();
#else
		float* w = icp.type->w.get();
		float* b = icp.type->biases.get();
#endif
		
			
		for (int i = 0; i < nl; i++) {
#ifdef CAUSAL_LOCAL_RULES
			accDestinationArray[i] = accDestinationArray[i] * (1.0f - kappa[i]) + destinationArray[i] * kappa[i];
#endif
			destinationArray[i] = b[i];
			for (int j = 0; j < nc; j++) {
				// += (H * alpha + w + wL) * prevAct
#ifdef CAUSAL_LOCAL_RULES
				destinationArray[i] += H[matID] * postSynActs[j];
#else
				destinationArray[i] += (H[matID] * alpha[matID] + w[matID] + wLifetime[matID]) * postSynActs[j];
#endif
				matID++;
			}
		}
	};

	auto hebbianUpdate = [this](InternalConnexion_P& icp, float* destinationArray) {
		int nl = icp.type->nLines;
		int nc = icp.type->nColumns;
		int matID = 0;

		float* A = icp.type->A.get();
		float* B = icp.type->B.get();
		float* C = icp.type->C.get();
		float* D = icp.type->D.get();
		float* eta = icp.type->eta.get();
		float* H = icp.H.get();
		float* E = icp.E.get();

#ifdef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* gamma = icp.type->gamma.get();
		float* alpha = icp.type->alpha.get();
#else
		float* avgH = icp.avgH.get();
#endif


#ifdef OJA
		float* delta = icp.type->delta.get();
#ifndef CONTINUOUS_LEARNING
		float* wLifetime = icp.wLifetime.get();
		float* alpha = icp.type->alpha.get();
#endif
#endif

#ifdef RANDOM_WB
		float* w = icp.w.get();
#else
		float* w = icp.type->w.get();
#endif

#ifdef CAUSAL_LOCAL_RULES
		float* accDestinationArray = icp.accPostSynActs.get();
#endif

		for (int i = 0; i < nl; i++) {
			for (int j = 0; j < nc; j++) {
#ifdef CONTINUOUS_LEARNING
				//wLifetime[matID] = (1 - gamma[matID]) * wLifetime[matID] + gamma[matID] * H[matID] * alpha[matID] * totalM[1]; // TODO remove ?
#endif
				E[matID] = (1.0f - eta[matID]) * E[matID] + eta[matID] *
#ifdef CAUSAL_LOCAL_RULES
					(postSynActs[j] - accPostSynActs[j]) * 
					(A[matID] * destinationArray[i] + B[matID] * accDestinationArray[i]) * 
					(1.0f + C[matID] * (w[matID] + alpha[matID] * H[matID] + wLifetime[matID]));
#else
					(A[matID] * destinationArray[i] * postSynActs[j] + B[matID] * destinationArray[i] + C[matID] * postSynActs[j] + D[matID]);
#endif

#ifdef OJA
				E[matID] -= eta[matID] * destinationArray[i] * destinationArray[i] * delta[matID] * (w[matID] + alpha[matID]*H[matID] + wLifetime[matID]);
#endif

				H[matID] += E[matID] * totalM[0];

#ifdef CAUSAL_LOCAL_RULES
				H[matID] = std::clamp(H[matID], -128.0f, 128.0f);
#else
				H[matID] = std::clamp(H[matID], -1.0f, 1.0f);
#endif

#ifndef CONTINUOUS_LEARNING
				avgH[matID] += H[matID];
#endif
				matID++;

			}
		}
	};

	auto applyNonLinearities = [firstCall](float* src, float* dst, ACTIVATION* fcts, int size
#ifdef STDP
		, float* acc_src, float* mu, float* lambda
#endif
		) 
	{

#ifdef CAUSAL_LOCAL_RULES
		float* accDestinationArray = icp.accPostSynActs.get();
		float* kappa = icp.type->kappa.get();
#endif

#ifdef STDP
		for (int i = 0; i < size; i++) {
			acc_src[i] = acc_src[i] * (1.0f-mu[i]) + src[i]; // * mu[i] ? TODO
		}
#else 
		float* acc_src = src;
#endif
		
		for (int i = 0; i < size; i++) {
			switch (fcts[i]) {
			case TANH:
				dst[i] = tanhf(acc_src[i]);
				break;
			case GAUSSIAN:
				dst[i] = 2.0f * expf(-std::clamp(powf(acc_src[i], 2.0f), -10.0f, 10.0f)) - 1.0f;
				break;
			case RELU:
				dst[i] = std::max(acc_src[i], 0.0f);
				break;
			case LOG2:
				dst[i] = std::max(log2f(abs(acc_src[i])), -100.0f);
				break;
			case EXP2:
				dst[i] = exp2f(std::clamp(acc_src[i], -30.0f, 30.0f));
				break;
			case SINE:
				dst[i] = sinf(acc_src[i]);
				break;
			case CENTERED_TANH:
				constexpr float z = 1.0f / .375261f; // to map to [-1, 1]
				dst[i] = tanhf(acc_src[i]) * expf(-std::clamp(powf(acc_src[i], 2.0f), -10.0f, 10.0f)) * z;
				break;
			}
		}

#ifdef STDP
		for (int i = 0; i < size; i++) {
			// acc_src's magnitude decreases when there is a significant activation.
			acc_src[i] -= lambda[i] * powf(dst[i], 2.0f * 1.0f + 1.0f); // TODO only works for tanh as of now
		}
#endif

#ifdef CAUSAL_LOCAL_RULES
		if (firstCall) [[unlikely]] {
			std::copy(destinationArray, destinationArray + nl, accDestinationArray);
		}
#endif
	};



	// STEP 1: MODULATION  A.
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize, 
			type->toModulation.activationFunctions.get(),
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->toModulation.STDP_mu.get(), type->toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = postSynActs[i + type->inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 2: MEMORY A.
	if (memoryChildren.size() != 0) {
		// Nothing is transmitted between this and the memory children, as their pointers
		// point towards the same data arrays.

		propagate(toMemory, memoryChildren[0].input);


		int activationFunctionId = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {

			// in place, as the memory child's input is shared with this node.
			applyNonLinearities(
				memoryChildren[i].input,
				memoryChildren[i].input,
				&type->toMemory.activationFunctions[activationFunctionId],
				memoryChildren[i].type->inputSize
#ifdef STDP
				, memoryChildren[i].accumulatedInput, &type->toMemory.STDP_mu[activationFunctionId], &type->toMemory.STDP_lambda[activationFunctionId]
#endif
			);

#ifdef SATURATION_PENALIZING
			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				float v = memoryChildren[i].input[j];
				*globalSaturationAccumulator += powf(v, saturationExponent);
				memoryChildren[i].saturationArray[j] += v;
			}
#endif

			activationFunctionId += memoryChildren[i].type->inputSize;
		}


		hebbianUpdate(toMemory, memoryChildren[0].input);


		for (int i = 0; i < memoryChildren.size(); i++) {
			memoryChildren[i].forward();
		}

	}


	// STEP 2: COMPLEX
	if (children.size() != 0) {
		float* ptrToInputs = preSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#ifdef STDP
		float* ptrToAccInputs = accumulatedPreSynActs + type->outputSize + MODULATION_VECTOR_SIZE;
#endif
		propagate(toChildren, ptrToInputs);
		
		

		// Apply non-linearities
		int id = 0;
		for (int i = 0; i < children.size(); i++) {


			applyNonLinearities(
				ptrToInputs + id,
				children[i].postSynActs,
				&type->toChildren.activationFunctions[id],
				children[i].type->inputSize
#ifdef STDP
				, ptrToAccInputs + id, &type->toChildren.STDP_mu[id], &type->toChildren.STDP_lambda[id]
#endif
			);

#ifdef SATURATION_PENALIZING 
			// child post-syn input
			int i0 = MODULATION_VECTOR_SIZE + id;
			for (int j = 0; j < children[i].type->inputSize; j++) {
				float v = children[i].postSynActs[j];
				*globalSaturationAccumulator += powf(abs(v), saturationExponent);
				averageActivation[i0+j] += v;
			}
#endif

			id += children[i].type->inputSize;
		}

		// has to happen after non linearities but before forward, 
		// for children's output not to have changed yet.
		hebbianUpdate(toChildren, ptrToInputs);


		// transmit modulation and apply forward, then retrieve the child's output.

		float* childOut = postSynActs + type->inputSize + MODULATION_VECTOR_SIZE;
		for (int i = 0; i < children.size(); i++) {

			for (int j = 0; j < MODULATION_VECTOR_SIZE; j++) {
				children[i].totalM[j] = this->totalM[j];
			}

			children[i].forward();

			std::copy(children[i].preSynActs, children[i].preSynActs + children[i].type->outputSize, childOut);
			childOut += children[i].type->outputSize;
		}

	}


	// STEP 4: MODULATION B.
	if (children.size() != 0 || memoryChildren.size() != 0)
	{
		propagate(toModulation, preSynActs + type->outputSize);
		applyNonLinearities(
			preSynActs + type->outputSize,
			postSynActs + type->inputSize,
			type->toModulation.activationFunctions.get(),
			MODULATION_VECTOR_SIZE
#ifdef STDP
			, accumulatedPreSynActs + type->outputSize, type->toModulation.STDP_mu.get(), type->toModulation.STDP_lambda.get()
#endif
		);
		hebbianUpdate(toModulation, postSynActs + type->inputSize);

		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			totalM[i] += postSynActs[i + type->inputSize];
		}

#ifdef SATURATION_PENALIZING 
		for (int i = 0; i < MODULATION_VECTOR_SIZE; i++) {
			float v = postSynActs[i + type->inputSize];
			*globalSaturationAccumulator += powf(abs(v), saturationExponent);
			averageActivation[i] += v;
		}
#endif
	}


	// STEP 5: MEMORY B.
	if (false && children.size() != 0 && memoryChildren.size() != 0) {
		// Nothing is transmitted between this and the memory children, as their pointers
		// point towards the same data arrays.

		propagate(toMemory, memoryChildren[0].input);


		int activationFunctionId = 0;
		for (int i = 0; i < memoryChildren.size(); i++) {

			// in place, as the memory child's input is shared with this node.
			applyNonLinearities(
				memoryChildren[i].input,
				memoryChildren[i].input,
				&type->toMemory.activationFunctions[activationFunctionId],
				memoryChildren[i].type->inputSize
#ifdef STDP
				, memoryChildren[i].accumulatedInput, &type->toMemory.STDP_mu[activationFunctionId], &type->toMemory.STDP_lambda[activationFunctionId]
#endif
			);

#ifdef SATURATION_PENALIZING
			for (int j = 0; j < memoryChildren[i].type->inputSize; j++) {
				float v = memoryChildren[i].input[j];
				*globalSaturationAccumulator += powf(v, saturationExponent);
				memoryChildren[i].saturationArray[j] += v;
			}
#endif

			activationFunctionId += memoryChildren[i].type->inputSize;
		}


		hebbianUpdate(toMemory, memoryChildren[0].input);


		for (int i = 0; i < memoryChildren.size(); i++) {
			memoryChildren[i].forward();
		}

	}


	// STEP 6: OUTPUT
	{
		propagate(toOutput, preSynActs);
		
		
		applyNonLinearities(
			preSynActs,
			preSynActs,
			type->toOutput.activationFunctions.get(),
			type->outputSize
#ifdef STDP
			, accumulatedPreSynActs, type->toOutput.STDP_mu.get(), type->toOutput.STDP_lambda.get()
#endif
		);

		hebbianUpdate(toOutput, preSynActs);
	}
	
}
