#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdlib>

#include "Network.hpp"
#include "Trainer.hpp"
#include "types.hpp"

int main () {
	srand(48123);

	Network nn{{2, 8, 1}};

	std::vector<Vector> inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
	std::vector<Vector> outputs = {{0}, {1}, {1}, {0}};

	Trainer t;

	for(int i = 0; i < 500; ++i){
		for(int j = 0; j < 4; ++j) {
			t.train(nn, inputs[j], outputs[j], 0.05);
		}
	}

	for(int j = 0; j < 4; ++j) {
		auto res = nn.predict(inputs[j]);
		std::cout << "GOT: " << res[0] << " EXPECTED: " << outputs[j][0] << '\n';
	}

}

