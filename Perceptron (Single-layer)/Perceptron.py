import numpy as np


weights = np.array([0.0, 0.0])  # initial weights for x1, x2
bias = 0.0  # initial bias
learning_rate = 0.1  # learning rate
epochs = 10  # number of iterations over the dataset

# Sample training data 
# Input format: [x1, x2], Output: [target]
training_data = [
    (np.array([0.5, 1.5]), 2.0),  # Input: [0.5, 1.5], Target: 2.0
    (np.array([1.0, 2.0]), 4.0),  # Input: [1.0, 2.0], Target: 4.0
    (np.array([1.5, 2.5]), 5.0),  # Input: [1.5, 2.5], Target: 5.0
    (np.array([2.0, 3.0]), 6.0)  # Input: [2.0, 3.0], Target: 6.0
]


# Activation function (step function)
def step_function(z):
    return z  # Identity function as we're working with continuous values


# Step 2: Training the perceptron
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")

    for inputs, target in training_data:
        # Step 3: Calculate the weighted sum (z)
        z = np.dot(inputs, weights) + bias

        # Step 4: Make prediction using the step function
        prediction = step_function(z)

        # Step 5: Update weights if prediction is wrong
        error = target - prediction

        # Update weights and bias
        weights += learning_rate * error * inputs
        bias += learning_rate * error

        # Print weight updates and prediction
        print(f"Inputs: {inputs}, Target: {target}, Prediction: {prediction}")
        print(f"Updated Weights: {weights}, Bias: {bias}\n")

# Final weights and bias after training
print(f"Final Weights: {weights}")
print(f"Final Bias: {bias}")

# Step 6: Test the perceptron with sample input
test_input = np.array([2.0, 3.0])  # Test input
test_output = step_function(np.dot(test_input, weights) + bias)
print(f"Prediction for input {test_input}: {test_output}")
