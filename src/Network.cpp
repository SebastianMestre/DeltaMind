// Network.cpp
#include "Network.hpp"

#include <cassert>
#include <cstdlib>

#include "math.hpp"

Network::Network (std::vector<int> sizes_) : sizes{std::move(sizes_)} {
	assert(1 < sizes.size());
	assert(0 < sizes.back());

	weights.resize(sizes.size() - 1);

	for (int i = 0; i < sizes.size() - 1; ++i) {
		assert(0 < sizes[i]);

		// add one to account for bias
		weights[i] = make_matrix(sizes[i+1], sizes[i]+1);

		for(int x = 0; x < weights[i].size(); ++x){
			for(int y = 0; y < weights[i][x].size(); ++y){
				weights[i][x][y] = ((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
			}
		}
	}
}

Vector Network::predict (Vector const& example_input) {
	assert(example_input.size() == sizes[0]);

	Vector current = example_input;

	for (int i = 0; i < sizes.size() - 1; ++i) {
		current.push_back(1.0f); // bias
		current = feed_forward(current, weights[i]);
	}

	current.pop_back();

	return current;
}
