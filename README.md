# Portfolio-Optimisation
Mean Variance Optimisation

- Log return distribution of the different assets
![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/a7de9e85-7725-4a6f-9020-8aedaab38f55)

- The portfolio is highly correlated
  
![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/c719131f-d5e5-47d2-8441-2990844b6b1a)

- According to the results on the test set, if we fear about any loose, it is not a good idea to do a long short strategy but in the other case, if we are a risk taker, long short is the best option. This result is quite different from what we could expect base on the training curve. There is kind of "overfitting" or "stability" issue. We have to correct it using a regularisation

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/9f558d77-56bb-4611-898b-9d3b3e6afcad)

- Stabilisation of the portfolio by ridge regression

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/876b7fd3-143b-47de-8517-a68d3a537798)

![image](https://github.com/MOMOJordan/Portfolio-Optimisation/assets/86100448/95949c43-8a77-4678-bfd6-4d81ff534ee4)


import numpy as np
import cvxpy as cp

# Parameters
n = 3  # dimension of each Wi
k = 4  # number of Wi vectors

# CVXPY variable
W = cp.Variable(n * k)

# Covariance blocks
Cov1 = np.eye(n)
Cov2 = 0.5 * np.ones((n, n))
big_cov = np.block([[Cov1, Cov2],
                    [Cov2, Cov1]])  # Shape: (2n, 2n)

# Build selector matrix S (shape: (k-1, k))
selector = np.zeros((k - 1, k))
for i in range(k - 1):
    selector[i, i] = 1
    selector[i, i + 1] = 1

# Expand with Kronecker product to create block selector
S = np.kron(selector, np.eye(n))  # Shape: ((k-1)*n, k*n)

# Build final quadratic form matrix
M = S.T @ np.kron(np.eye(k - 1), big_cov) @ S  # Shape: (k*n, k*n)

# Final CVXPY expression
expr = cp.quad_form(W, M)

