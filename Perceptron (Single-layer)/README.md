# Perceptron

A Perceptron is the simplest type of artificial neuron which works in a similar way of biological neurons. We can call this the grandfather of all neural networks.  
It takes multiple inputs, applies weights, adds a bias, and outputs a decision — usually binary (like yes/no, 1/0, spam/not spam).

The structure of a perceptron will look something like shown below :

x1 -----\  
x2 ------> (Weighted Sum) --> Activation --> Output (0 or 1)  
x3 -----/

Lets formulate the mathematical equation:  
Let x1,x2,x3...xn be the input parameters and w1,w2,w3...wn be the weights  
The weighted sum can be represented by the equation -  
z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + b  
where b is the bias.

Please keep in mind that this z is nothing but a linear combination of input parameters with their weights and perceptrons can be applied only in linearly separable data.  
output = Activation(z)

Now lets discuss more about activation function

Most single perceptron learners use a step function also known as Heaviside step function.
              
output = ( 1, if z >= 0 // 0, if z < 0 )
            

This is what makes it a classifier. It separates inputs into two categories.  
Say for example if you are trying to classify a mail as spam or not spam,then if the value of z is greater than 0 then its a spam else not spam.

### How does the weight gets adjusted ?

Well, Perceptron learns using a simple Perceptron Learning Rule.

Below are the steps in a Perceptron learning:

Step 1: Initialize weights and bias (random or zero).  
Step 2: For each training example calculate output.  
Step 3: Compare with actual label.  
Step 4: Update weights if prediction is wrong.  
Step 5: Weight Update Rule: 

wi = wi + α * (target - prediction) * xi
b = b + α * (target - prediction)


Where:

- `wi` = Current weight for input `xi`  
  The goal of training is to adjust these weights so that predictions get more accurate.  
  Each input `xi` has an associated weight `wi`.

- `α` = Learning Rate (hyper-parameter)  
  A small constant (e.g., 0.01 or 0.1) that controls how big each update step is.  
  If `α` is too high, updates might overshoot the target; too low, and training becomes slow.

- `(target - prediction)` = Error  
  This is the difference between actual output (label) and what the model predicted.  
  Possible values:  
    +1 → Model predicted too low, needs to increase weight  
    0  → Model predicted correctly, no update needed  
    -1 → Model predicted too high, needs to reduce weight

- `xi` = Input feature  
  The weight update is proportional to the input value.  
  Inputs with higher values will have a stronger influence on weight change.

Let's do a quick math with some random numbers:

- `wi = 0.5` (initial weight)  
- `xi = 1` (input value)  
- `prediction = 0`  
- `target = 1`  
- `α = 0.1`

Now plugging these numbers to our equation gives:

wi = 0.5 + 0.1 * (1 - 0) * 1
= 0.5 + 0.1
= 0.6


Initial weight was 0.5 and now its 0.6.  
The weight increased a bit — that’s the model learning.

Similarly we need to calculate the same for bias:
b = b + 0.1 * (1 - 0) = b + 0.1


Think about this way: The bias shifts the decision boundary, like moving the entire line left or right (in 2D visualization).

### So What Happens if you dont adjust or never use Bias?

Let’s say `b = 0`, so the equation becomes:

w1x1 + w2x2 = 0


We all know that the general equation of a line is `y = mx + c` and `c` is the intercept. Now lets say our line is like shown below. Lets take `c = 5` as the intercept
```
 y-axis ↑
    |
 10 |             
  9 |             
  8 |             
  7 |           x  
  6 |        x     
  5 |-----x--------- ← this is the intercept (bias)
  4 |     x     o  
  3 |   x     o    
  2 |     o     o  
  1 |        o      
  0 |     o         
     ----------------→ x-axis
        |    |    |
```
Now, if there is no bias then `c` has to be 0, the line must pass through the origin (0,0).

That’s a huge limitation — because what if the best boundary does NOT pass through the origin?

No matter how you tune the weights, the line is stuck pivoting around the origin, which might not allow correct classification at all.

These points are perfectly linearly separable — but only if the line doesn’t pass through origin.

If you set `b = 0`, you might never get a clean separation, no matter how you rotate the line using weights.
```
 y-axis ↑
        |
 10 |           x
  9 |         x  
  8 |       x    
  7 |     x     
  6 |   x      
  5 |           
  4 |     x     o  
  3 |   x     o    
  2 |     o     o  
  1 |        o      
  0 |--x------------- ← this is the intercept now (bias = 0)
     ----------------→ x-axis
```
### Bias Gives You More Flexibility

It's like giving the model the freedom to place the line anywhere on the plane, not just rotate it.

Weights define direction, bias defines position.

Together, they make the perceptron powerful enough to classify a wide range of linearly separable patterns.

