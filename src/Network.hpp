// Network.hpp
#pragma once

#include "types.hpp"

struct Network {
	friend class Trainer;

	std::vector<int> sizes;
	std::vector<Matrix> weights;

	Network (std::vector<int> sizes_);

	Vector predict (Vector const& example_input);
};
