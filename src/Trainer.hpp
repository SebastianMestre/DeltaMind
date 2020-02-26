// Trainer.hpp
#pragma once

#include "Network.hpp"
#include "types.hpp"

struct Trainer {
	void train (
		Network& network,
		Vector const& example_input,
		Vector const& example_output,
		float learning_rate
	);
};
