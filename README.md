# -ANN-Fundamentals-
# 1. Introduction & Architecture:
This readme documents the design and results of a Multi-Layer Perceptron (MLP) neural network built entirely from scratch using NumPy — no TensorFlow, no PyTorch. The network classifies Iris flowers into three species (Setosa, Versicolor, Virginica) using four features: sepal length, sepal width, petal length, and petal width (150 total samples). The goal was to build a deep understanding of the core mathematics behind neural networks.

<img width="787" height="147" alt="image" src="https://github.com/user-attachments/assets/87d7925e-487c-42d6-8c30-ab89b18105f9" />

The decreasing layer size (8 → 6) creates a bottleneck that forces the network to compress and abstract information, reducing overfitting on this small dataset. Total trainable parameters: 107.

# 2.  Implementation:
2.1 Data Preparation-

The dataset was split 70/15/15 (train/val/test) using stratified sampling. Features were normalized using Z-score: z = (x − μ) / σ, fitted only on training data to prevent leakage. Labels were one-hot encoded ([1,0,0], [0,1,0], [0,0,1]) to match the softmax output format. 

2.2 Forward Propagation-

Data flows layer-by-layer. Each layer applies a linear transformation followed by a non-linear activation:
     Z⁽ˡ⁾ = A⁽ˡ⁻¹⁾ · W⁽ˡ⁾ + b⁽ˡ⁾    →    A⁽ˡ⁾ = activation(Z⁽ˡ⁾)
Intermediate Z and A values are cached during the forward pass because backpropagation needs them to compute gradients. The output layer uses Softmax to produce class probabilities that sum to 1.0. 

2.3 Loss Function-

Cross-entropy loss measures the gap between predicted probabilities and true labels. A small epsilon (1e-8) prevents log(0). L2 regularization (λ=0.001) penalizes large weights to reduce overfitting:
     L = −(1/N) × Σ y_true × log(ŷ + ε)  +  (λ/2N) × Σ‖W‖² 
     
2.4 Backpropagation — Chain Rule-

Gradients flow backwards from output to input. The key insight is that the combined derivative of Softmax + Cross-Entropy simplifies beautifully to just (prediction − truth):
     Output layer :  ∂L/∂Z³ = A³ − y_true
     Hidden layers:  ∂L/∂Z² = (∂L/∂A²) ⊙ σ'(Z²)   where σ'(z) = σ(z)(1−σ(z))
     Weights      :  ∂L/∂W² = (A¹)ᵀ · ∂L/∂Z² / N    Biases: ∂L/∂b = Σ(∂L/∂Z) / N
     
The ⊙ symbol means element-wise multiplication. The gradient is then passed backward to the previous layer as: ∂L/∂A¹ = ∂L/∂Z² · (W²)ᵀ 

2.5 Training Loop-

Mini-batch gradient descent was used with batch size 16 — a balance between the noise of stochastic (1 sample) and the slow computation of full-batch gradient descent. Training data is shuffled each epoch. Weights update: W = W − α × ∂L/∂W. Early stopping halts training if validation accuracy shows no improvement for 150 epochs, restoring the best weights found.

# 3. Results & Experiments:
3.1 Main Model Performance-

<img width="802" height="103" alt="image" src="https://github.com/user-attachments/assets/83e626ce-1223-4620-8626-78077e7cb5cf" />

Our from-scratch implementation matches sklearn's optimized MLPClassifier, validating the correctness of every function — from weight initialization to the backward pass. Setosa achieves near-perfect F1 (1.00) as it is linearly separable. Versicolor and Virginica, which overlap in feature space, still achieve F1 > 0.90.

3.2 Hyperparameter Experiments-

<img width="805" height="489" alt="image" src="https://github.com/user-attachments/assets/346cd693-ed53-4d97-ab98-d988a1777c28" />

3.3 Key Findings- 

•	Learning rate = 0.01 is the sweet spot. Too low (0.001) → slow. Too high (0.1) → diverges.

•	Tanh slightly outperforms Sigmoid because it is zero-centered; ReLU converges fastest early.

•	Deeper/wider networks do NOT help on small datasets — the default 107-parameter network is sufficient.

•	L2 regularization (λ=0.001) and early stopping effectively prevent overfitting.

•	Adam optimizer (β₁=0.9, β₂=0.999) converges faster and needs less learning rate tuning than SGD.

#  4. Conclusions:

This assignment achieved its primary goal: building a working, accurate neural network from mathematical first principles using only NumPy. The model exceeds the 90% test accuracy target and matches sklearn's production implementation. More importantly, implementing every operation by hand — matrix multiplications, the sigmoid and its derivative, the chain rule layer-by-layer, mini-batch shuffling, and bias-corrected Adam moments — builds an intuition that no framework tutorial can replicate.

The experiments reveal that hyperparameter choices matter significantly. A poorly chosen learning rate alone can drop accuracy by 8–10 percentage points. Activation function choice matters less on this dataset but becomes critical in deeper networks. Network capacity should match dataset size — bigger is not always better.

