// The derivative of the loss function with respect to y_hat: dL/dy_hat
float dL_dy_hat(float y, float y_hat);

// The derivative of the output of the neuron with respect to the weighted sum:
// dy_hat/dz
// which is the derivative of the sigmoid function
float dy_hat_dz(float y_hat);

// The derivative of the weighted sum with respect to the k-th weight: dz/dWk
float dz_dWk(float input);

// The derivative of the loss function with respect to the k-th weight: dL/dWk
// which is the product of the above derivatives
// dL/dWk = dL/dy_hat * dy_hat/dz * dz/dWk
float dL_dWk(float y, float y_hat, float input);