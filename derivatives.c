/*
 * This file contains the implementations of the derivatives that will be used
   in the backpropagation to adjust the weights of our neuron.

 * Using the chain rule, we can calculate the derivative of the loss function
   with respect to the weights of the neuron as:
   	___________________________________________
   	| dL/dWk = dL/dy_hat * dy_hat/dz * dz/dWk |
    	-------------------------------------------
   where:
   	* L is the loss function.
    	* y_hat is the output of the neuron
     	* z is the weighted sum of the inputs.
      	* Wk is the k-th weight of the neuron.
 */

// The derivative of the loss function with respect to y_hat: dL/dy_hat
float dL_dy_hat(float y, float y_hat)
{
	// Calculate the derivative of the loss function with respect to y_hat
	return (y_hat - y) / (y_hat - (y_hat * y_hat));
}

// The derivative of the output of the neuron with respect to the weighted sum:
// dy_hat/dz
// which is the derivative of the sigmoid function
float dy_hat_dz(float y_hat)
{
	return y_hat * (1 - y_hat);
}

// The derivative of the weighted sum with respect to the k-th weight: dz/dWk
float dz_dWk(float input)
{
	return input;
}

// The derivative of the loss function with respect to the k-th weight: dL/dWk
// which is the product of the above derivatives
// dL/dWk = dL/dy_hat * dy_hat/dz * dz/dWk
float dL_dWk(float y, float y_hat, float input)
{
	return dL_dy_hat(y, y_hat) * dy_hat_dz(y_hat) * dz_dWk(input);
}