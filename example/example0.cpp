#include <iostream>
#include <vector>

#include <cstdlib>

#include "Network.hpp"
#include "Trainer.hpp"
#include "types.hpp"

/*
 * This program illustrates the usage of the library by training a neural net
 * to calculate xor.
 */
int main () {
	srand(48123);

	/*
	 * When we construct a Network, we pass the size of each layer in a
	 * std::vector<int>.
	 */
	Network nn{{2, 8, 1}};

	/*
	 * Inputs and outputs for training are also expressed as std::vectors
	 */
	std::vector<Vector> inputs  = {{0,0}, {0,1}, {1,0}, {1,1}};
	std::vector<Vector> outputs = {{0},   {1},   {1},   {0}};

	/*
	 * To train a network, we need to instance a trainer.
	 */
	Trainer t;

	std::cout << "Training...";
	for(int i = 0; i < 5000; ++i){
		for(int j = 0; j < 4; ++j) {
			/*
			 * We can call train with the appropiate arguments.
			 * This will update our network in-place.
			 */
			t.train(nn, inputs[j], outputs[j], 0.05);
		}
	}
	std::cout << "done.\n";

	for(int j = 0; j < 4; ++j) {
		/*
		 * To get values out of the network, we use predict
		 */
		auto res = nn.predict(inputs[j]);
		std::cout << "GOT: " << res[0] << " EXPECTED: " << outputs[j][0] << '\n';
	}
}

