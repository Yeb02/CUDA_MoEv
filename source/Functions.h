#pragma once

#include <vector>

int binarySearch(std::vector<float>& proba, float value);

int binarySearch(float* proba, float value, int size);

// Normal mutation in the space of log(half-life constant).
// m default value is .15f set in this file's header.
float mutateDecayParam(float dp, float m = .15f);

// src is unchanged. src can be the same array as dst.
// Dst has mean 0 and variance 1.
void normalizeArray(float* src, float* dst, int size);

// src is unchanged. src can be the same array as dst.
// Dst values in [-1, 1], -1 attibuted to the worst of src and 1 to the best.
void rankArray(float* src, float* dst, int size);

