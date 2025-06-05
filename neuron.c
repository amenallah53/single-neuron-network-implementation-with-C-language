#include "neuron.h"
#include <stdlib.h>
#include <time.h>

// Create a new neuron with a specified activation function
Neuron *new_neuron(float (*activation_function)(float x))
{
	// Allocate memory for a new neuron
	Neuron *neuron = (Neuron *)malloc(sizeof(Neuron));

	// Check if the memory allocation was successful
	if (neuron == NULL) {
		// Return NULL if the memory allocation failed
		return NULL;
	}

	// Set the weights to random values between -1 and 1
	srand(time(NULL));
	for (int i = 0; i < 2; i++) {
		neuron->weights[i] = (float)rand() / RAND_MAX * 2 - 1;
	}

	// Set the bias to 0
	neuron->bias = 0;

	// Set the activation function of the neuron
	neuron->activation_function = activation_function;

	// Return a pointer to the newly created neuron if everything was successful
	return neuron;
}

// Runs the neuron with a given input
float run_neuron(Neuron *neuron, float *input)
{
	// Calculate the weighted sum of the inputs
	float weighted_sum = 0;
	for (int i = 0; i < 2; i++) {
		weighted_sum += neuron->weights[i] * input[i];
	}

	// Add the bias to the weighted sum
	weighted_sum += neuron->bias;

	// Return the result of the activation function
	return neuron->activation_function(weighted_sum);
}

// Frees the memory allocated for a neuron
void free_neuron(Neuron *neuron)
{
	// Free the memory allocated for the neuron
	free(neuron);
}