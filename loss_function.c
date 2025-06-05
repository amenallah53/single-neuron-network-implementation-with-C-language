/*
 * This file defines the loss function that we will use in this project which is
   Cross Entropy Loss.

 * The loss value is a measure of how well our neuron is performing on a given
   dataset, and it is used to update its weights during training.

 * The cross entropy loss function is defined as:
 	___________________________________________________________
  	| L = -1/N * Σ(y * log(y_hat) + (1 - y) * log(1 - y_hat)) |
   	-----------------------------------------------------------
   where N is the number of samples in the dataset, y is the true label of the
   sample, y_hat is the predicted label of the sample, and log is the natural
   logarithm.
 */

#include <math.h>

// Calculate the loss value of a neuron on a given dataset using the Cross
// Entropy Loss function.
float cross_entropy_loss(float *y, float *y_hat, int N)
{
	// Initialize the loss value
	float loss = 0;

	// Calculate the loss value
	for (int i = 0; i < N; i++) {
		loss += y[i] * log(y_hat[i]) + (1 - y[i]) * log(1 - y_hat[i]);
	}

	// Return the average loss value
	return -1.0 / N * loss;
}