// math.hpp
#pragma once

#include <utility>

#include "types.hpp"

/*
 * Neuron activation function
 */
float activation (float x);

/*
 * Derivative of activation composed with inverse of activation
 */
float derivative_comp_inverse (float x);

/*
 * A single step in the feed forward algorithm
 */
Vector feed_forward (Vector const& input, Matrix const& weights);

/*
 * A single step in the back propagation algorithm
 *
 * Returns gradients of cost wrt:
 * - The current layer of neurons
 * - The weights between that and the next layer
 */
std::pair<Vector, Matrix> propagate_gradient (
	Vector const& current, Vector const& next,
	Matrix const& weights, Vector const& dC_dnext);

/*
 * Returns an N x M matrix
 * first index is in [0,N)
 */
Matrix make_matrix (int N, int M);
