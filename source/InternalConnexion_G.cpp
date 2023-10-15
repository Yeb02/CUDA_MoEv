#pragma once

#include "InternalConnexion_G.h"


float InternalConnexion_G::decayParametersInitialValue = -1.0f;


InternalConnexion_G::InternalConnexion_G(int nRows, int nColumns) :
	nRows(nRows), nColumns(nColumns)
{

	int s = nRows * nColumns;
	if (s == 0) return;

	float f0 = 1.0f;
	f0 = powf((float)nColumns, -.5f); 
	
	auto rand = [&s](float* vec, float b, float f) {
		for (int i = 0; i < s; i++) {
			vec[i] = NORMAL_01 * f + b;
		}
	};

	auto rand01 = [&s](float* vec) {
		for (int i = 0; i < s; i++) {
			vec[i] = mutateDecayParam(decayParametersInitialValue, .4f); // Not default m for better initial spread.
		}
	};


	storage = std::make_unique<float[]>(getNParameters());

	createArraysFromStorage();

	float* _storagePtr = storage.get();

	
	for (int i = 0; i < N_STATIC_MATRICES_01; i++) 
	{
		rand01(_storagePtr);
		_storagePtr += s;
	}

	for (int i = 0; i < N_STATIC_MATRICES_R; i++)
	{
		rand(_storagePtr, 0.0f, f0);
		_storagePtr += s;
	}

	s = nRows;

	for (int i = 0; i < N_STATIC_VECTORS_01; i++)
	{
		rand01(_storagePtr);
		_storagePtr += s;
	}

	for (int i = 0; i < N_STATIC_VECTORS_R; i++)
	{
		rand(_storagePtr, 0.0f, f0);
		_storagePtr += s;
	}

}


InternalConnexion_G::InternalConnexion_G(const InternalConnexion_G& gc)
{
	nRows = gc.nRows;
	nColumns = gc.nColumns;

	int s = getNParameters();

	storage = std::make_unique<float[]>(s);
	std::copy(gc.storage.get(), gc.storage.get() + s, storage.get());

	createArraysFromStorage();
}


InternalConnexion_G InternalConnexion_G::operator=(const InternalConnexion_G& gc) {
	nRows = gc.nRows;
	nColumns = gc.nColumns;

	int s = getNParameters();

	storage = std::make_unique<float[]>(s);
	std::copy(gc.storage.get(), gc.storage.get() + s, storage.get());

	createArraysFromStorage();

	return *this;
}


void InternalConnexion_G::mutate(float p) {

	//param(t+1) = (b+a*N1)*param(t) + c*N2
	//const float sigma = powf((float)nColumns, -.5f);
	const float sigma = 1.0F;
	const float a = .3f * sigma;
	const float b = 1.0f-a*.3f;
	const float c = a;


	int size = nRows * nColumns;
	SET_BINOMIAL(size, p);

	auto mutateMatrix = [&size, p, a, b, c](float* matrix)
	{

		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(size);


			matrix[matrixID] *= b + NORMAL_01 * a;
			matrix[matrixID] += NORMAL_01 * c;
		}
	};

	auto mutateDecayMatrix = [&size, p](float* matrix)
	{

		int _nMutations = BINOMIAL;
		for (int k = 0; k < _nMutations; k++) {
			int matrixID = INT_0X(size);
			matrix[matrixID] = mutateDecayParam(matrix[matrixID]);
		}
	};
	
	for (int i = 0; i < N_STATIC_MATRICES_01; i++)
	{
		mutateDecayMatrix(matrices01[i].data());
	}

	for (int i = 0; i < N_STATIC_MATRICES_R; i++)
	{
		mutateMatrix(matricesR[i].data());
	}

	size = nRows;
	SET_BINOMIAL(size, p);

	for (int i = 0; i < N_STATIC_VECTORS_01; i++)
	{
		mutateDecayMatrix(vectors01[i].data());
	}

	for (int i = 0; i < N_STATIC_VECTORS_R; i++)
	{
		mutateDecayMatrix(vectorsR[i].data());
	}

}


InternalConnexion_G::InternalConnexion_G(std::ifstream& is)
{
	READ_4B(nRows, is);
	READ_4B(nColumns, is);

	storage = std::make_unique<float[]>(getNParameters());
	is.read(reinterpret_cast<char*>(storage.get()), getNParameters() * sizeof(float));

	createArraysFromStorage();
}


void InternalConnexion_G::save(std::ofstream& os)
{
	WRITE_4B(nRows, os);
	WRITE_4B(nColumns, os);

	os.write(reinterpret_cast<const char*>(storage.get()), getNParameters() * sizeof(float));
}


void InternalConnexion_G::createArraysFromStorage()
{
	int s = nRows * nColumns;

	float* _storagePtr = storage.get();

	matrices01.reserve(N_STATIC_MATRICES_01);
	for (int i = 0; i < N_STATIC_MATRICES_01; i++)
	{
		matrices01.emplace_back(_storagePtr, nRows, nColumns);
		_storagePtr += s;
	}

	matricesR.reserve(N_STATIC_MATRICES_R);
	for (int i = 0; i < N_STATIC_MATRICES_R; i++)
	{
		matricesR.emplace_back(_storagePtr, nRows, nColumns);
		_storagePtr += s;
	}

	s = nRows;

	vectors01.reserve(N_STATIC_VECTORS_01);
	for (int i = 0; i < N_STATIC_VECTORS_01; i++)
	{
		vectors01.emplace_back(_storagePtr, nRows);
		_storagePtr += s;
	}

	vectorsR.reserve(N_STATIC_VECTORS_R);
	for (int i = 0; i < N_STATIC_VECTORS_R; i++)
	{
		vectorsR.emplace_back(_storagePtr, nRows);
		_storagePtr += s;
	}
}
