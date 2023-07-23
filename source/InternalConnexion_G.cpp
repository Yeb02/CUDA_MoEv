#pragma once

#include "InternalConnexion_G.h"



// Normal mutation in the space of log(half-life constant). m default value is .15f.
inline float mutateDecayParam(float dp, float m) 
{
	float exp_r = exp2f(NORMAL_01 * m);
	float tau = log2f(1.0f - dp);


	float t = 1.0f - exp2f(exp_r * tau);


	if (t < 0.001f) {
		t = 0.001f;
	}
	else if (t > .999f) {
		t = .999f;
	}

	return 1.0f - exp2f(exp_r * tau);
}



InternalConnexion_G::InternalConnexion_G(int nLines, int nColumns) :
	nLines(nLines), nColumns(nColumns)
{

	int s = nLines * nColumns;

	float f0 = 1.0f;
	if (s != 0) { f0 = powf((float)nColumns, -.5f); }
	 

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);


	auto rand = [&s](float* vec, float b, float f) {
		for (int i = 0; i < s; i++) {
			vec[i] = NORMAL_01 * f + b;
		}
	};

	auto rand01 = [&s](float* vec) {
		for (int i = 0; i < s; i++) {
			vec[i] = mutateDecayParam(DECAY_PARAMETERS_INIT_BIAS, .4f); // Not default m for better initial spread.
		}
	};

	rand(A.get(), 0.0f, f0);
	rand(B.get(), 0.0f, f0);
	rand(C.get(), 0.0f, f0);
	rand01(eta.get());

	s = nLines;

	kappa = std::make_unique<float[]>(s);
	rand01(kappa.get());

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	rand01(STDP_mu.get());
	rand01(STDP_lambda.get());
#endif
}

InternalConnexion_G::InternalConnexion_G(const InternalConnexion_G& gc) {
	
	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);
	

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());


	s = nLines;
	
	kappa = std::make_unique<float[]>(s);
	std::copy(gc.kappa.get(), gc.kappa.get() + s, kappa.get());

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
#endif
}

InternalConnexion_G InternalConnexion_G::operator=(const InternalConnexion_G& gc) {

	nLines = gc.nLines;
	nColumns = gc.nColumns;

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	A = std::make_unique<float[]>(s);
	B = std::make_unique<float[]>(s);
	C = std::make_unique<float[]>(s);

	std::copy(gc.eta.get(), gc.eta.get() + s, eta.get());
	std::copy(gc.A.get(), gc.A.get() + s, A.get());
	std::copy(gc.B.get(), gc.B.get() + s, B.get());
	std::copy(gc.C.get(), gc.C.get() + s, C.get());


	s = nLines;
	
	kappa = std::make_unique<float[]>(s);
	std::copy(gc.kappa.get(), gc.kappa.get() + s, kappa.get());

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	std::copy(gc.STDP_mu.get(), gc.STDP_mu.get() + s, STDP_mu.get());
	std::copy(gc.STDP_lambda.get(), gc.STDP_lambda.get() + s, STDP_lambda.get());
#endif

	return *this;
}

void InternalConnexion_G::mutateFloats(float p) {

	//param(t+1) = (b+a*N1)*param(t) + c*N2
	//const float sigma = powf((float)nColumns, -.5f);
	const float sigma = 1.0F;
	const float a = .3f * sigma;
	const float b = 1.0f-a*.3f;
	const float c = a;

#ifdef GUIDED_MUTATIONS
	// w += clip[-accumulatorClipRange,accumulatorClipRange](accumulator)
	constexpr float accumulatorClipRange = 1.0f;
#endif

	int size = nLines * nColumns;
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
	
	mutateMatrix(A.get());
	mutateMatrix(B.get());
	mutateMatrix(C.get());

	mutateDecayMatrix(eta.get());


	size = nLines;
	SET_BINOMIAL(size, p);

	mutateDecayMatrix(kappa.get());

#ifdef STDP
	mutateDecayMatrix(STDP_lambda.get());
	mutateDecayMatrix(STDP_mu.get());
#endif

}

InternalConnexion_G::InternalConnexion_G(std::ifstream& is)
{
	READ_4B(nLines, is);
	READ_4B(nColumns, is);

	int s = nLines * nColumns;

	eta = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(eta.get()), s * sizeof(float));
	A = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(A.get()), s * sizeof(float));
	B = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(B.get()), s * sizeof(float));
	C = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(C.get()), s * sizeof(float));

	s = nLines;

	kappa = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(kappa.get()), s * sizeof(float));

#ifdef STDP
	STDP_mu = std::make_unique<float[]>(s);
	STDP_lambda = std::make_unique<float[]>(s);
	is.read(reinterpret_cast<char*>(STDP_mu.get()), s * sizeof(float));
	is.read(reinterpret_cast<char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}

void InternalConnexion_G::save(std::ofstream& os)
{
	WRITE_4B(nLines, os);
	WRITE_4B(nColumns, os);

	int s = nLines * nColumns;

	os.write(reinterpret_cast<const char*>(eta.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(A.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(B.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(C.get()), s * sizeof(float));

	s = nLines;

	os.write(reinterpret_cast<const char*>(kappa.get()), s * sizeof(float));

#ifdef STDP
	os.write(reinterpret_cast<const char*>(STDP_mu.get()), s * sizeof(float));
	os.write(reinterpret_cast<const char*>(STDP_lambda.get()), s * sizeof(float));
#endif
}
