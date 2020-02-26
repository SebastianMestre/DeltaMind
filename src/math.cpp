// math.cpp
#include "math.hpp"

#include "types.hpp"

float activation (float x) {
	return x < 0.0f ? 0.1f * x : x;
}

float derivative_comp_inverse (float x) {
	return x < 0.0f ? 0.1 : 1.0f;
}

Vector feed_forward (Vector const& input, Matrix const& weights) {
	Vector output(weights.size(), 0.0f);
	
	for (int i = 0; i < output.size(); ++i) {
		for (int j = 0; j < input.size(); ++j) {
			output[i] += input[j] * weights[i][j];
		}
		output[i] = activation(output[i]);
	}

	return output;
}

std::pair<Vector, Matrix> propagate_gradient (
	Vector const& current,
	Vector const& next,
	Matrix const& weights,
	Vector const& dC_dnext
) {
	Vector dC_dcurrent(current.size(), 0.0f); // gradient of values
	Matrix dC_dweights(weights.size(), Vector(current.size(), 0.0f)); // gradient of weights

	for (int i = 0; i < weights.size(); ++i) {
		const float dcinext_times_dcdnext = derivative_comp_inverse(next[i]) * dC_dnext[i];

		for (int j = 0; j < current.size(); ++j) {
			dC_dcurrent[j] += dcinext_times_dcdnext * weights[i][j];
			dC_dweights[i][j] = dcinext_times_dcdnext * current[j];
		}
	}

	return {dC_dcurrent, dC_dweights};
}

Matrix make_matrix (int N, int M) {
	return Matrix(N, Vector(M, 0.0f));
}
