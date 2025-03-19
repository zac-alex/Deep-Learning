# Activation Functions

The activation function is a key mathematical function used in neural networks to decide if a neuron should be activated or not. It takes the weighted sum of the neuron’s inputs, processes them, and calculates a new value that determines how much signal gets passed on to the next layer in the network. In simpler terms, it controls how strongly a neuron reacts to the input values.

This function is essential for training neural networks because it helps the network model complex, non-linear relationships. Choosing the right activation function for the network architecture and the specific data can greatly impact the performance and outcomes of the model, making it a critical part of building a neural network.

A linear function can be thought of as a basic activation function that simply multiplies the input by a constant 
c, such as 𝑐×𝑥. When c=1, it becomes the identity function. However, this linear function doesn’t introduce any non-linearity to the neural network.

![Example Image 1](Images/Figure_2.png)

Non-linearity is essential in neural networks because, without it, even networks with multiple layers would only produce linear outputs, no matter how many layers are added. Since most real-world data is not linearly separable, adding non-linear layers helps transform the data in a way that allows the network to learn more complex patterns and use various objective functions effectively.

![Example Image 2](Images/Figure_1.png)

Now if you look into the above plot,you can see that there is no way you can seperate two classes A & B using a straight line.In other words this data is not linearly seperable.
That is exactly where activation functions comes into play.
There are two main properties for an activation function:

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
