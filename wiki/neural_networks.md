# Introduction to Neural Networks: Theory

## Table of Contents

1. [Biological Inspiration: The Neuron](#11-biological-inspiration-the-neuron)
2. [The McCulloch-Pitts Neuron (1943)](#12-the-mcculloch-pitts-neuron-1943)
3. [Linear Separability](#13-linear-separability)
4. [Rosenblatt's Perceptron (1957)](#14-rosenblatts-perceptron-1957)
5. [The XOR Problem](#15-the-xor-problem)
6. [Minsky and Papert's Critique (1969)](#16-minsky-and-paperts-critique-1969)
7. [The Solution: Multi-Layer Networks](#17-the-solution-multi-layer-networks)
8. [Activation Functions](#18-activation-functions)
9. [The Multi-Layer Perceptron (1986)](#19-the-multi-layer-perceptron-1986)

---

## 1.1 Biological Inspiration: The Neuron

Artificial neural networks are inspired by the biological neurons in the human brain. Understanding this connection helps grasp the design of artificial models.

**Biological Neuron Components:**

- **Dendrites**: Receive signals from other neurons
- **Soma (Cell Body)**: Processes the incoming information
- **Axon**: Transmits the output signal to other neurons
- **Synapse**: Point of connection between neurons, where signal transmission occurs

**Key insight:** A biological neuron receives multiple input signals, integrates them, and "fires" (produces an output) only if the combined signal exceeds a threshold. This all-or-nothing behavior inspired the first artificial neuron models.

**Mathematical abstraction:**

$$\text{output} = f\left(\sum_{i=1}^{n} w_i x_i - \theta\right)$$

where:
- $x_i$ are input signals
- $w_i$ are connection strengths (weights)
- $\theta$ is the firing threshold
- $f$ is an activation function

---

## 1.2 The McCulloch-Pitts Neuron (1943)

Warren McCulloch and Walter Pitts proposed the first mathematical model of a neuron in 1943, laying the groundwork for neural computation.

**Model definition:**

The M-P neuron takes binary inputs $x_1, x_2, \ldots, x_n \in \{0, 1\}$ and produces a binary output $y \in \{0, 1\}$:

$$y = H\left(\sum_{i=1}^{n} x_i - \theta\right)$$

where $H$ is the **Heaviside step function**:

$$H(z) = \begin{cases} 1, & z \geq 0 \\ 0, & z < 0 \end{cases}$$

**Key properties:**
- All inputs are binary (0 or 1)
- All weights are equal (fixed at 1)
- Threshold $\theta$ must be set manually
- Output is binary (fire or don't fire)


**Implementing logical operators:**

The M-P neuron can implement basic Boolean functions by choosing appropriate thresholds:

**AND operator** (with $n$ inputs):
$$y = H\left(\sum_{i=1}^{n} x_i - n\right)$$

The neuron fires only when ALL inputs are 1.

| $x_1$ | $x_2$ | $\sum x_i$ | $\sum x_i - 2$ | $y$ |
|-------|-------|------------|----------------|-----|
| 0 | 0 | 0 | -2 | 0 |
| 0 | 1 | 1 | -1 | 0 |
| 1 | 0 | 1 | -1 | 0 |
| 1 | 1 | 2 | 0 | 1 |

**OR operator** (with $n$ inputs):
$$y = H\left(\sum_{i=1}^{n} x_i - 1\right)$$

The neuron fires when ANY input is 1.

| $x_1$ | $x_2$ | $\sum x_i$ | $\sum x_i - 1$ | $y$ |
|-------|-------|------------|----------------|-----|
| 0 | 0 | 0 | -1 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 2 | 1 | 1 |

**NOT operator** (single input):
$$y = H(-x_1 + 0.5) = H(0.5 - x_1)$$

This requires an inhibitory connection (negative weight), extending the original model.


**Limitations of McCulloch-Pitts:**
1. Only accepts binary inputs
2. All weights are equal (no learning)
3. Threshold must be manually specified
4. Cannot handle non-linearly separable functions

---

## 1.3 Linear Separability

A fundamental concept in understanding neural network capabilities is **linear separability**.

**Definition:** A Boolean function is **linearly separable** if there exists a hyperplane that separates the inputs that produce output 1 from those that produce output 0.

In 2D, this means we can draw a straight line to separate the two classes.

**Geometric interpretation:**

For a neuron with two inputs, the decision boundary is:
$$w_1 x_1 + w_2 x_2 = \theta$$

This defines a line in 2D space. The neuron outputs 1 on one side of the line and 0 on the other.

**AND is linearly separable:**
```
x₂
 ↑
 1 |  ○     ●      ← (1,1) is the only 1
   |
 0 |  ○     ○
   +-------→ x₁
     0     1

A line can separate ● from ○
```

**OR is linearly separable:**
```
x₂
 ↑
 1 |  ●     ●
   |
 0 |  ○     ●
   +-------→ x₁
     0     1

A line can separate ● from ○
```

**Why linear separability matters:**
Single-layer neural networks (M-P neurons, perceptrons) can ONLY learn linearly separable functions. This is their fundamental limitation.

---

## 1.4 Rosenblatt's Perceptron (1957)

Frank Rosenblatt introduced the perceptron, extending the M-P neuron with trainable weights.

**Model definition:**

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^T \mathbf{x} + b)$$

where:
- $x_i$ can be real numbers (not just binary)
- $w_i$ are learnable weights
- $b$ is a learnable bias (replaces threshold: $b = -\theta$)
- $f$ is the activation function

**Improvements over McCulloch-Pitts:**
1. **Real-valued inputs**: Not limited to binary
2. **Trainable weights**: Each input has its own adjustable weight
3. **Learnable bias**: Threshold is learned, not manually set
4. **Learning algorithm**: Weights can be automatically adjusted

**The Perceptron Learning Algorithm:**

```
Initialize: w = 0, b = 0
Repeat until convergence:
    For each training example (x, y_target):
        y_pred = sign(w·x + b)
        if y_pred ≠ y_target:
            w = w + η · y_target · x
            b = b + η · y_target
```

where $\eta$ is the learning rate.

**Key insight:** The algorithm adjusts weights to reduce classification errors. If the prediction is wrong, it moves the decision boundary toward correctly classifying the misclassified point.

**Perceptron Convergence Theorem:**
If the training data is linearly separable, the perceptron learning algorithm is guaranteed to converge to a solution in a finite number of steps.

**Vectorized form:**

For computational efficiency, we can absorb the bias into the weight vector:

$$\tilde{\mathbf{x}} = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad \tilde{\mathbf{w}} = \begin{bmatrix} b \\ w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}$$

Then: $y = f(\tilde{\mathbf{w}}^T \tilde{\mathbf{x}})$

---

## 1.5 The XOR Problem

The **XOR (exclusive OR)** function exposed a critical limitation of single-layer networks.

**XOR Truth Table:**

| $x_1$ | $x_2$ | $x_1 \oplus x_2$ |
|-------|-------|------------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Why XOR is not linearly separable:**

```
x₂
 ↑
 1 |  ●     ○
   |
 0 |  ○     ●
   +-------→ x₁
     0     1

No single line can separate ● from ○!
```

The 1s and 0s are arranged diagonally. Any straight line you draw will misclassify at least one point.

**Mathematical proof:**

Assume a perceptron can compute XOR. Then there exist $w_1, w_2, b$ such that:

$$\begin{align}
w_1(0) + w_2(0) + b &< 0 \quad \text{(output 0 for input 0,0)} \\
w_1(0) + w_2(1) + b &\geq 0 \quad \text{(output 1 for input 0,1)} \\
w_1(1) + w_2(0) + b &\geq 0 \quad \text{(output 1 for input 1,0)} \\
w_1(1) + w_2(1) + b &< 0 \quad \text{(output 0 for input 1,1)}
\end{align}$$

From equations (1) and (4): $b < 0$ and $w_1 + w_2 + b < 0$

From equations (2) and (3): $w_2 + b \geq 0$ and $w_1 + b \geq 0$

Adding (2) and (3): $w_1 + w_2 + 2b \geq 0$

But from (1): $b < 0$, so $w_1 + w_2 + 2b < w_1 + w_2 + b$

This means: $w_1 + w_2 + b > w_1 + w_2 + 2b \geq 0$

But equation (4) requires $w_1 + w_2 + b < 0$. **Contradiction!**

Therefore, no single perceptron can compute XOR.

---

## 1.6 Minsky and Papert's Critique (1969)

In their influential book "Perceptrons," Marvin Minsky and Seymour Papert rigorously analyzed the limitations of single-layer perceptrons.

**Key findings:**
1. Single perceptrons cannot solve XOR
2. Single perceptrons cannot recognize many patterns
3. The book (mistakenly) implied these limitations extended to multi-layer networks

**The "AI Winter":**
This critique led to reduced funding and interest in neural network research for nearly 15 years. This period is often called the "AI Winter."

**What they missed:**
Minsky and Papert's analysis focused on single-layer networks. Multi-layer networks with appropriate training algorithms CAN solve XOR and many other non-linearly separable problems. However, at the time, there was no known efficient algorithm to train multi-layer networks.

---

## 1.7 The Solution: Multi-Layer Networks

The XOR problem can be solved by combining multiple perceptrons into layers.

**Key insight:** XOR can be expressed as a combination of linearly separable functions:

$$\text{XOR}(x_1, x_2) = (x_1 \text{ OR } x_2) \text{ AND NOT}(x_1 \text{ AND } x_2)$$

Or equivalently:
$$\text{XOR}(x_1, x_2) = (x_1 \text{ AND NOT } x_2) \text{ OR } (\text{NOT } x_1 \text{ AND } x_2)$$

**Two-layer XOR network:**

```
Input Layer    Hidden Layer    Output Layer

   x₁ ─────┬────→ [h₁] ─────┐
           ╳                 ├───→ [out] ───→ y
   x₂ ─────┴────→ [h₂] ─────┘
```

**One possible solution:**

Hidden unit 1 (computes $x_1$ AND $x_2$):
$$h_1 = H(x_1 + x_2 - 1.5)$$

Hidden unit 2 (computes $x_1$ OR $x_2$):
$$h_2 = H(x_1 + x_2 - 0.5)$$

Output unit (computes $h_2$ AND NOT $h_1$):
$$y = H(h_2 - h_1 - 0.5) = H(-h_1 + h_2 - 0.5)$$

**Verification:**

| $x_1$ | $x_2$ | $h_1$ (AND) | $h_2$ (OR) | $y$ (XOR) |
|-------|-------|-------------|------------|-----------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

**Why this works:**
The hidden layer transforms the input space into a new representation where XOR becomes linearly separable. This is the fundamental power of multi-layer networks.

---

## 1.8 Activation Functions

The step function (Heaviside) has a critical problem: it's not differentiable. This makes it impossible to use gradient-based learning algorithms.

**Smooth activation functions:**

**Sigmoid (Logistic) function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties:
- Output range: $(0, 1)$
- Continuous and differentiable
- Derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Useful for binary classification (output interpretable as probability)

**Disadvantage:** Vanishing gradient for large $|z|$

**Hyperbolic tangent (tanh):**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

Properties:
- Output range: $(-1, 1)$
- Zero-centered (often preferred over sigmoid)
- Derivative: $\tanh'(z) = 1 - \tanh^2(z)$

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(z) = \max(0, z)$$

Properties:
- Output range: $[0, \infty)$
- Computationally efficient
- Helps with vanishing gradient problem
- Most popular in modern deep learning

**Why differentiability matters:**
To train multi-layer networks, we need to compute how the error changes with respect to each weight. This requires taking derivatives through the entire network (backpropagation). Non-differentiable activation functions break this chain.

---

## 1.9 The Multi-Layer Perceptron (1986)

The breakthrough came with the **backpropagation algorithm**, popularized by Rumelhart, Hinton, and Williams in 1986.

**Network architecture:**

A multi-layer perceptron (MLP) consists of:
- **Input layer**: Receives the features
- **Hidden layer(s)**: Perform intermediate computations
- **Output layer**: Produces the final prediction

**Forward propagation:**

For a 2-layer network (one hidden layer):

$$\begin{align}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{a}^{(1)} &= g(\mathbf{z}^{(1)}) \\
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)} \\
\hat{y} &= g(\mathbf{z}^{(2)})
\end{align}$$

where:
- $\mathbf{W}^{(l)}$ is the weight matrix for layer $l$
- $\mathbf{b}^{(l)}$ is the bias vector for layer $l$
- $g$ is the activation function
- $\mathbf{a}^{(l)}$ is the activation (output) of layer $l$

**The Loss Function:**

For regression (MSE):
$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2$$

For binary classification (Binary Cross-Entropy):
$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

**Training via Gradient Descent:**

Update rule:
$$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$$

The challenge: computing $\nabla_\theta J$ for weights in hidden layers.

**Backpropagation** solves this by applying the chain rule systematically from output to input layers, propagating error signals backward through the network.

---

## Key Formulas Summary

**McCulloch-Pitts Neuron:**
$$y = H\left(\sum_{i=1}^{n} x_i - \theta\right)$$

**Perceptron:**
$$y = f(\mathbf{w}^T \mathbf{x} + b)$$

**Weight update:**
$$\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$$

**Sigmoid:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Binary Cross-Entropy:**
$$J = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})\right]$$

**Gradient Descent:**
$$\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)$$

---

## References

**Books**:
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning* (2009)
- Bishop. *Pattern Recognition and Machine Learning* (2006)
- Goodfellow, Bengio, Courville. *Deep Learning* (2016)

**Historical Papers**:
- McCulloch & Pitts. "A Logical Calculus of Ideas Immanent in Nervous Activity" (1943)
- Rosenblatt. "The Perceptron: A Probabilistic Model for Information Storage" (1958)
- Minsky & Papert. "Perceptrons" (1969)
- Rumelhart, Hinton, Williams. "Learning Representations by Back-Propagating Errors" (1986)
---

*This notes basis on WUST Machine Learning tutorial by [dr Daniel Kucharczyk](https://dkucharc.github.io/academic/).*
