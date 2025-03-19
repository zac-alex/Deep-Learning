The activation function is a key mathematical function used in neural networks to decide if a neuron should be activated or not. It takes the weighted sum of the neuron’s inputs, processes them, and calculates a new value that determines how much signal gets passed on to the next layer in the network. In simpler terms, it controls how strongly a neuron reacts to the input values.

This function is essential for training neural networks because it helps the network model complex, non-linear relationships. Choosing the right activation function for the network architecture and the specific data can greatly impact the performance and outcomes of the model, making it a critical part of building a neural network.

A linear function can be thought of as a basic activation function that simply multiplies the input by a constant 
c, such as 𝑐×𝑥. When c=1, it becomes the identity function. However, this linear function doesn’t introduce any non-linearity to the neural network. Non-linearity is essential in neural networks because, without it, even networks with multiple layers would only produce linear outputs, no matter how many layers are added. Since most real-world data is not linearly separable, adding non-linear layers helps transform the data in a way that allows the network to learn more complex patterns and use various objective functions effectively.

![Example Image](images/Figure_1.png)
