#include "neuron.h"

// Train the neuron using backpropagation
int train(Neuron *neuron, float **inputs, float *outputs, int samples_num,
	  float learning_rate, int num_epoch);
