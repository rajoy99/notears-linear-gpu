Notears PyTorch

A PyTorch implementation of the NOTEARS algorithm (Non-parametric Optimization for Structure Learning) for causal discovery.

This package provides a continuous optimization approach to learning DAGs (Directed Acyclic Graphs) from data, harnessing the power of GPUs to accelerate the learning process.

Note: The original NOTEARS implementation is CPU only.

📦 Installation

You can install this package directly from PyPi:

pip install notears-pytorch


🚀 Usage

Below is a basic example of how to use the linear NOTEARS algorithm.


from notears_pytorch import notears_linear

# 1. Generate or load your data
# X = np.random.randn(n_samples, d_nodes) ...

# 2. Run optimization
# Returns a binary adjacency matrix where B[i, j] = 1 implies i -> j
adj_matrix = notears_linear(X, lambda1=0.1, w_threshold=0.3)

print("Estimated Adjacency Matrix:")
print(adj_matrix)


📚 API

notears_linear(X, lambda1=0.1, ...)

Solves the optimization problem to find the DAG structure.

X (np.ndarray):
The data matrix of shape (n, d), where n is the number of samples and d is the number of nodes/variables.

lambda1 (float):
L1 penalty parameter. Controls the sparsity of the resulting graph.

rho_init (float):
Initial value for the augmented Lagrangian penalty parameter.

w_threshold (float):
Threshold for edge weights. Edges with an absolute weight below this value are pruned.

use_gpu (bool):
If True and CUDA is available, computations will be performed on the GPU for faster processing.

📄 Citation

If you use this implementation in your research, please cite this GitHub repository and the original paper:

Zheng, X., Aragam, B., Ravikumar, P. K., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. Advances in Neural Information Processing Systems.