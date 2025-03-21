# Activation Functions

The activation function is a key mathematical function used in neural networks to decide if a neuron should be activated or not. It takes the weighted sum of the neuron’s inputs, processes them, and calculates a new value that determines how much signal gets passed on to the next layer in the network. In simpler terms, it controls how strongly a neuron reacts to the input values.

This function is essential for training neural networks because it helps the network model complex, non-linear relationships. Choosing the right activation function for the network architecture and the specific data can greatly impact the performance and outcomes of the model, making it a critical part of building a neural network.

# Linear Activation Function

A linear function can be thought of as a basic activation function that simply multiplies the input by a constant 
c, such as 𝑐×𝑥. When c=1, it becomes the identity function. However, this linear function doesn’t introduce any non-linearity to the neural network.

![Example Image 1](Images/Figure_2.png)

# Non-Linear Activation Function

Non-linearity is essential in neural networks because, without it, even networks with multiple layers would only produce linear outputs, no matter how many layers are added. Since most real-world data is not linearly separable, adding non-linear layers helps transform the data in a way that allows the network to learn more complex patterns and use various objective functions effectively.

![Example Image 2](Images/Figure_1.png)

Now if you look into the above plot,you can see that there is no way you can seperate two classes A & B using a straight line.In other words this data is not linearly seperable.
That is exactly where activation functions comes into play.
There are two main properties for an activation function in a neural network:

1.Non-linearity (discussed above)

2.Differentiable

For a neural network to learn, its activation functions must be differentiable. 
This means that the function must have a derivative, which allows us to calculate 
how much the function's output changes in response to small changes in the input. 
This is important because the neural network learns through a process called 
backpropagation, where the network adjusts its internal settings like the weights 
to improve predictions.

Here's how it works in simple terms: During training, the network makes a prediction, 
then compares it to the actual result. The difference (or error) is used to 
adjust the weights. To figure out how to adjust them, we calculate the gradient 
(the derivative) of the activation function, which tells us how sensitive the output 
is to changes in the input.

Let’s say we have a very simple network with just one neuron and use the sigmoid 
activation function:

   σ(x) = 1 / (1 + e^(-x))

For example, if the input x = 2, then:

    σ(2) = 1 / (1 + e^(-2)) ≈ 0.88
    
Now, we need to know how to adjust the weights based on the error. For that, we 
need the derivative of the sigmoid function:

    σ'(x) = σ(x)(1 - σ(x))

For x = 2, this gives:

    σ'(2) = 0.88 * (1 - 0.88) ≈ 0.105

This derivative tells us how much the output changes if we tweak the input slightly. 
During backpropagation, we use this information to adjust the weights to improve the 
prediction, and by repeating this process, the network learns over time.

In short, the differentiability of the activation function allows the neural network 
to adjust its weights and improve its predictions step by step.

#
Activation functions have many advantages, but they can also have challenges during training. Some activation functions, like Sigmoid or Tanh, have areas where their gradients become very small, often approaching zero. These areas are called saturation regions. In these regions, small changes in the input values cause very tiny changes in the output of the activation function. As a result, the training process slows down significantly. This is known as the vanishing gradient problem.

For example, consider the Sigmoid activation function:

    σ(x) = 1 / (1 + e^(-x))

When the input value is very large or very small (in the range where the sigmoid function approaches its maximum or minimum), the gradient (or derivative) becomes almost zero. For instance:

    For x = 10, σ(10) ≈ 0.99995 (close to 1)
    The derivative σ'(x) ≈ 0.00005 (a very small value)

Now, if we want to adjust the weights based on this small gradient, the update becomes extremely small. This means the network learns very slowly in these regions, leading to slower convergence during training.

![Example Image 3](Images/Figure_3.png)

This issue is known as the Vanishing Gradient Problem. It occurs when the gradients (derivatives) of activation functions become very small, particularly in regions where the activation function is saturated (e.g., near the maximum or minimum values of the Sigmoid or Tanh functions). This leads to very small weight updates during backpropagation, slowing down or even halting the learning process in deep neural networks.
In simpler terms, when the values of the input push the activation function to its extremes (near the max or min), the function doesn't change much anymore. This slow response can hinder the learning process, making it harder for the network to adjust and improve.

# Logistic Sigmoid & Tanh Activation Function

In the early days of neural networks, two common activation functions (AFs) used to introduce non-linearity were the **Logistic Sigmoid** and **Tanh** functions. These functions were inspired by the firing of biological neurons in the human brain.

#### Logistic Sigmoid Function:
The **Logistic Sigmoid** is a very popular and traditional activation function. It's mathematically defined as:

Logistic Sigmoid(x) = 1 / (1 + e^(-x))

This function takes an input `x` and squashes it into a range between 0 and 1. So, no matter how large or small the input is, the output will always stay between 0 and 1.

This squashing effect helps neural networks model complex patterns in the data, but it comes with the problem vanishing gradient problem.

#### Tanh Function

To address the vanishing gradient problem and other issues with the Logistic Sigmoid, the **Tanh** (hyperbolic tangent) function is often used. The Tanh function has a similar shape to the sigmoid but has a key difference: its output is zero-centered. This means that the output values range from -1 to 1, instead of 0 to 1.

The Tanh function is given by:

Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

It is also non-linear and squashes the input, but it has the advantage of being zero-centered. This makes the training process more stable and can help with convergence.

#### Zero-Centered Nature of Tanh:

The **Tanh function** is **zero-centered**, which is an important property. This means that the output values of the Tanh function range from **-1 to 1** instead of **0 to 1** like the Logistic Sigmoid. Being zero-centered has significant benefits:

1. **Helps with convergence**: When the data is zero-centered, it helps the optimization process. Neural networks rely on gradient-based optimization techniques (like stochastic gradient descent) to update the weights during training. If the output of neurons is mostly positive (like in the case of the sigmoid function), the updates to weights can lead to slow learning because the gradients will be skewed in one direction. With Tanh, the zero-centered nature ensures that the gradients can flow more evenly, making learning more stable and faster.
   
2. **Reduces bias**: The zero-centered nature of Tanh also means that the activations can be negative, which provides more flexibility to the model. It helps the network explore a broader set of possibilities instead of being restricted to positive outputs.

##### Example:
Let's say `x = 2`:

Tanh(2) = (e^2 - e^(-2)) / (e^2 + e^(-2)) ≈ 0.96  (This values is in between -1 and 1)

#### Why Zero-Centered Is Helpful:
Consider what happens when the inputs to the neurons are large positive or large negative values. For the **Logistic Sigmoid** function, if the input is large positive, the output will approach 1, and if the input is large negative, the output will approach 0. This creates an issue because during backpropagation, the gradient of the sigmoid function near its saturation points (near 0 or 1) becomes very small. This means the network's learning will slow down because the weights won't get updated effectively.

Both sigmoid and tanh can suffer from the vanishing gradient problem in deep networks, where gradients become very small and stop the weights from updating effectively. However, tanh is generally preferred because it has a wider range of gradient values (compared to sigmoid), making it more effective in practice for learning over longer time periods and deeper networks.

![Example Image 4](Images/Figure_5.png)

Both sigmoid and tanh can suffer from the vanishing gradient problem in deep networks, where gradients become very small and stop the weights from updating effectively. However, tanh is generally preferred because it has a wider range of gradient values (compared to sigmoid), making it more effective in practice for learning over longer time periods and deeper networks.

###Why tanh isn't directly used for probabilities:

Sigmoid is specifically designed for probability-like outputs because it maps any input to a value between 0 and 1. This is why it's typically used in binary classification tasks where the output represents a probability of class membership.

Tanh actually maps input to values between -1 and 1. This makes it more suited for tasks where the output needs to be centered around zero, such as in regression problems or hidden layers of neural networks. Tanh's range allows for both positive and negative activations, which can be beneficial in modeling complex relationships in data.

Lets assume a network with 1 input layer,1 hidden layer with Tanh activation function and 1 output layer with sigmoid activation function.

**What does the Hidden layer do?**:

The **tanh** activation function maps the input to values between **-1 and 1**. This helps the network learn both positive and negative features, making it better at capturing complex relationships in the data or in other words
the hidden layer transforms the input in a way that the network can learn from both positive and negative patterns.

**Output Layer with sigmoid**:

The **sigmoid** activation function maps the output to values between **0 and 1**, making it perfect for situations where you want the output to represent a **probability** and can be used for
for binary classification tasks.

Let’s say we have a simple neural network with:
- **Input value** = 0.5
- **Weight between input and hidden layer** = 2
- **Bias for hidden layer** = 0.1
- **Weight between hidden layer and output layer** = -1.5
- **Bias for output layer** = 0.2

### Hidden Layer Calculation (Using tanh):

First, we will calculate the input to the hidden neuron by multiplying the input by the weight and adding the bias:

z1 = (w1 * x) + b1 = (2 * 0.5) + 0.1 = 1.0 + 0.1 = 1.1

Now, we apply the tanh function to this value:

hidden_output = tanh(1.1) ≈ 0.7616

So, the output of the hidden layer is 0.7616.

### Output Layer Calculation (Using sigmoid):

Now, we take the output of the hidden layer (which is 0.7616) and pass it through the output layer. First, we calculate the input to the output neuron:

z2 = (w2 * hidden_output) + b2 = (-1.5 * 0.7616) + 0.2 = -1.1424 + 0.2 = -0.9424

Finally, we apply the sigmoid function to this value:

output = 1 / (1 + e^(-z2)) = 1 / (1 + e^(0.9424)) ≈ 1 / (1 + 2.567) ≈ 0.279

The final output (after applying sigmoid) is approximately 0.279. This means that the model predicts a 27.9% chance that the input belongs to class 1 (or class 0).

---

### How can tanh be used in the output layer if it's bound to -1 and 1?

We typically wouldn't use tanh in the output layer for most real-world problems where the output needs to be either an unrestricted continuous value or a probability. This is because tanh is bounded between -1 and 1, which limits its applicability, especially in cases like regression tasks or classification tasks that require output values beyond that range, such as probabilities between 0 and 1.
