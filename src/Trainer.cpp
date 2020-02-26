// Trainer.cpp
#include "Trainer.hpp"

#include <algorithm>

#include <cassert>

#include "Network.hpp"
#include "types.hpp"
#include "math.hpp"

void Trainer::train (
	Network& network,
	Vector const& example_input,
	Vector const& example_output,
	float learning_rate
) {
	assert(example_input.size() == network.sizes[0]);

	std::vector<Vector> results;

	Vector current = example_input;
	current.push_back(1.0f); // bias

	results.push_back(current);
	for (int i = 0; i < network.weights.size(); ++i) {
		current = feed_forward(current, network.weights[i]);

		current.push_back(1.0f); // bias
		results.push_back(current);
	}

	Vector dC_dO(network.sizes.back()+1);
	dC_dO.back() = 0.0f;
	for (int i = 0; i < network.sizes.back(); ++i) {
		dC_dO[i] = (results.back()[i] - example_output[i]);
	}

	std::vector<Matrix> weights_gradients;

	Vector current_gradient = dC_dO;
	for (int l = network.weights.size(); l--;) {

		auto new_gradients = propagate_gradient(
				results[l], results[l+1],
				network.weights[l], current_gradient
				);

		weights_gradients.push_back(new_gradients.second);

		current_gradient = new_gradients.first;
	}

	std::reverse(weights_gradients.begin(), weights_gradients.end());

	for (int l = 0; l < network.weights.size(); ++l) {
		for (int i = 0; i < network.weights[l].size(); ++i) {
			for (int j = 0; j < network.weights[l][i].size(); ++j) {
				float delta = weights_gradients[l][i][j] * learning_rate;
				network.weights[l][i][j] -= delta;
			}
		}
	}
}
