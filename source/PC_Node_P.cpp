#pragma once

#include "MoEvCore.h"
#ifdef PREDICTIVE_CODING

#include "PC_Node_P.h"


PC_Node_P::PC_Node_P(PC_Node_G* _type, PC_Node_G** nodes, int i, int iC, int* nC, int tNC) :
	type(_type),
	toChildren(),
	toOutput(&_type->toOutput),
	inCoutActivations(nullptr, 0), outputActivations(nullptr, 0), inputActivations(nullptr, 0),
	inCoutAccumulators(nullptr, 0), outputAccumulators(nullptr, 0), inputAccumulators(nullptr, 0)
{
	tNC *= nC[0];
	
	children.reserve(nC[0]);
	toChildren.reserve(nC[0]);

	for (int j = 0; j < nC[0]; j++) {
		int child_i = nC[0] * i + j;
		children.emplace_back(nodes[iC + child_i], nodes, child_i, iC + tNC, nC+1, tNC);
		toChildren.emplace_back(&_type->toChildren[j]);
	}
};


PC_Node_P::PC_Node_P(const PC_Node_P& pcnp) :
	type(pcnp.type),
	toChildren(),
	toOutput(pcnp.toOutput),
	inCoutActivations(nullptr, 0), outputActivations(nullptr, 0), inputActivations(nullptr, 0),
	inCoutAccumulators(nullptr, 0), outputAccumulators(nullptr, 0), inputAccumulators(nullptr, 0)
{
	children.reserve(type->nChildren);
	toChildren.reserve(type->nChildren);

	for (int j = 0; j < type->nChildren; j++) 
	{	
		children.emplace_back(pcnp.children[j]);
		toChildren.emplace_back(pcnp.toChildren[j]);
	}
}

void PC_Node_P::setArrayPointers(float** ptr_activations, float** ptr_accumulators, float* outActivations, float* outAccumulators)
{

	// placement new is the recommended way: https://eigen.tuxfamily.org/dox/classEigen_1_1Map.html

	// (toOutput and toChildren's matrices have the same number of columns, we could have used either)
	new (&inCoutActivations) MVector(*ptr_activations, toOutput.type->nColumns);
	new (&inCoutAccumulators) MVector(*ptr_accumulators, toOutput.type->nColumns);

	new (&inputActivations) MVector(*ptr_activations, type->inputSize);
	new (&inputAccumulators) MVector(*ptr_accumulators, type->inputSize);

	new (&outputActivations) MVector(outActivations, type->outputSize);
	new (&outputAccumulators) MVector(outAccumulators, type->outputSize);

	int offset = (type->nChildren == 0 ? 0 : type->nChildren * children[0].type->outputSize) + type->inputSize;
	*ptr_activations  += offset;
	*ptr_accumulators += offset;
	
	
	float* actCoutPtr = inCoutActivations.data() + type->inputSize;
	float* accCoutPtr = inCoutAccumulators.data() + type->inputSize;
	for (int i = 0; i < children.size(); i++) {
		children[i].setArrayPointers(ptr_activations, ptr_accumulators, actCoutPtr, accCoutPtr);
		actCoutPtr += children[i].type->outputSize;// same for all i.
		accCoutPtr += children[i].type->outputSize;
	}
}


void PC_Node_P::xUpdate_simultaneous()
{
	// The update of an activation consists in substracting the gradient of the energy * learning rate. This (-)gradient is accumulated in the _Accumulators arrays,
	// and is the difference of 2 numbers: the epsilon and the eta.
	// 
	// epsilon_l = X_l - bias_l+1 + theta_l+1*f(x_l+1)
	// eta_l     = (theta_l-1.transposed * epsilon_l-1) * f'(x_l)
	// -grad = eta - epsilon
	// x -= lr * grad
	//  
	// "This" handles the update of its children's inputs and outputs. If this is the root node, the output and input of this are managed by the PC_Network.



	// grad_out = - epsilon_out = bias + theta*f(X_inCout) - X_out
	outputAccumulators += (toOutput.vectors[0] + toOutput.matrices[0] * inCoutActivations.array().tanh().matrix()) - outputActivations;

	// grad_inCout += eta_inCout = (thetaTransposed * epsilon_out) * f'(X_inCout)
	inCoutAccumulators -= ((toOutput.matrices[0].transpose() * outputAccumulators).array() * (1.0f - inCoutActivations.array().tanh().square())).matrix();

	for (int i = 0; i < type->nChildren; i++) 
	{
		// grad_Cin = - epsilon_Cin = bias + theta*f(X_inCout) - X_Cin
		children[i].inputAccumulators.noalias() = (toChildren[i].vectors[0] + toChildren[i].matrices[0] * inCoutActivations.array().tanh().matrix()) - children[i].inputActivations;

		// grad_inCout += eta_inCout = (thetaTransposed * epsilon_Cin) * f'(X_inCout)
		inCoutAccumulators -= ((toChildren[i].matrices[0].transpose() * children[i].inputAccumulators).array() * (1.0f - inCoutActivations.array().tanh().square())).matrix();


		// child's update
		children[i].xUpdate_simultaneous();

		children[i].inputActivations += .5f * children[i].inputAccumulators;
		children[i].outputActivations += .5f * children[i].outputAccumulators;
	}

	return;
}


void PC_Node_P::thetaUpdate_simultaneous() 
{
	const float lr = .0002f;

	// stored in the grad accumulator for convenience: f(X_inCout)
	inCoutAccumulators.noalias() = inCoutActivations.array().tanh().matrix();


	// stored in the grad accumulator for convenience: epsilon_out = X_out - (theta*f(X_inCout) + bias)
	outputAccumulators.noalias() = outputActivations - (toOutput.vectors[0] + toOutput.matrices[0] * inCoutAccumulators);

	// lr is injected here to do n multiplications instead of n*(n+1) 
	outputAccumulators *= lr;

	// TODO tester avec + et -

	// theta -= theta_grad * lr (theta_grad = f(x_inCout) * epsilon_out.transpose)
	toOutput.matrices[0] -= outputAccumulators * inCoutAccumulators.transpose();


	// bias -= bias_grad * lr (bias_grad = epsilon_out)
	toOutput.vectors[0] -= outputAccumulators;


	for (int i = 0; i < type->nChildren; i++)
	{
		// stored in the grad accumulator for convenience: epsilon_cIn = X_cIn - (theta*f(X_inCout) + bias)
		children[i].inputAccumulators.noalias() = children[i].inputActivations - (toChildren[i].vectors[0] + toChildren[i].matrices[0] * inCoutAccumulators);

		// lr is injected here to do n multiplications instead of n*(n+1) 
		children[i].inputAccumulators *= lr;

		// theta -= theta_grad * lr (theta_grad = f(x_inCout) * epsilon_cIn.transpose)
		toChildren[i].matrices[0] -= children[i].inputAccumulators * inCoutAccumulators.transpose();

		// bias -= bias_grad * lr (bias_grad = epsilon_cIn)
		toChildren[i].vectors[0] -= children[i].inputAccumulators;
		


		// child's update
		children[i].thetaUpdate_simultaneous();
	}


	return;
}


#endif