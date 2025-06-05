#include "neuron.h"
#include "activation_functions.h"
#include "train.h"
#include <stdio.h>

int main()
{
	// Create a new neuron with the sigmoid activation function
	Neuron *neuron = new_neuron(sigmoid);

	FILE* file = fopen("dataset.csv", "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file.\n");
        return 1; // Return an error code
    }

    // Variables for the data to be read from the CSV
    float x[100][2];
    int y[100];

    // Read the first line (header) to ignore it
    char header[1024];
    if (fgets(header, sizeof(header), file) == NULL) {
        fprintf(stderr, "Error: Unable to read the file.\n");
        fclose(file);
        return 1;
    }

    int i = 0;
    while (feof(file) != 1){
        // Read the data line
        fscanf(file, "%f,%f,%d", &x[i][0], &x[i][1], &y[i]);
        i++;
    }

    // Close the file
    fclose(file);

	//train
	train(neuron,&x[0][0],&y,i-1,0.1,200);
	

	// Free the memory allocated for the neuron
	free_neuron(neuron);

	return 0;
}
