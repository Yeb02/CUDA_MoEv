#include "Node_G.h"

Node_G::Node_G(int* inS, int* outS, int nC) :
	inputSize(inS[0]), outputSize(outS[0]), nChildren(nC),
	toModulation(MODULATION_VECTOR_SIZE, computeNCols(inS, outS, nC)),
	toChildren(nC > 0 ? nC * inS[1] : 0, computeNCols(inS, outS, nC)),
	toOutput(outS[0], computeNCols(inS, outS, nC))
{
	
};

Node_G::Node_G(Node_G* n) {

	inputSize = n->inputSize;
	outputSize = n->outputSize;

	toChildren = n->toChildren;
	toModulation = n->toModulation;
	toOutput = n->toOutput;

	children.reserve(n->children.size());
	for (int j = 0; j < n->children.size(); j++) {
		children.emplace_back(n->children[j]);
	}
}


Node_G::Node_G(std::ifstream& is) {

	READ_4B(inputSize, is);
	READ_4B(outputSize, is);

	int _s;
	READ_4B(_s, is);
	children.resize(_s);
	
	toChildren = InternalConnexion_G(is);
	toModulation = InternalConnexion_G(is);
	toOutput = InternalConnexion_G(is);

}

void Node_G::save(std::ofstream& os) {
	WRITE_4B(inputSize, os);
	WRITE_4B(outputSize, os);

	int _s;
	_s = (int)children.size();
	WRITE_4B(_s, os);
	
	toChildren.save(os);
	toModulation.save(os);
	toOutput.save(os);
}

void Node_G::mutateFloats(float adjustedFMutationP)
{
	toChildren.mutateFloats(adjustedFMutationP);
	toModulation.mutateFloats(adjustedFMutationP);
	toOutput.mutateFloats(adjustedFMutationP);
}


