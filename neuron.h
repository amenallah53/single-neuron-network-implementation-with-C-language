#ifndef NEURON
#define NEURON

// Define the structure of a neuron
typedef struct {
	float weights[2];
	float bias;
	float (*activation_function)(float x);
} Neuron;

// Define the constructor for the neuron
Neuron *new_neuron(float (*activation_function)(float x));

// Define the function to run the neuron
float run_neuron(Neuron *neuron, float *inputs);

// Define the function to free the neuron from memory
void free_neuron(Neuron *neuron);

#endif
