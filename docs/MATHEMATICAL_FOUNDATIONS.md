# Mathematical foundations: why this demonstrates multivariable calculus

## 1. The optimization problem

For n eligible ETFs, the forecast return vector is \(\hat\mu\in\mathbb R^n\), the estimated covariance matrix is symmetric \(\Sigma\in\mathbb R^{n\times n}\), and \(w\in\mathbb R^n\) are long-only allocation weights.

\[
\min_w f(w)=\tfrac\lambda2\sum_{i=1}^n\sum_{j=1}^n w_i\Sigma_{ij}w_j-\sum_{i=1}^n \hat\mu_iw_i,
\quad\text{s.t. } \sum_{i=1}^n w_i=1,\quad 0\le w_i\le c.
\]

Here \(c=0.5\), and \(n\cdot c\ge 1\) is required for feasibility. For the default portfolio, n=3.

## 2. First and second partial derivatives

By symmetry of covariance, differentiating with respect to the kth portfolio weight yields:

\[
\frac{\partial f}{\partial w_k}=\lambda\sum_{j=1}^n\Sigma_{kj}w_j-\hat\mu_k.
\]

This is the gradient \(\nabla f=\lambda\Sigma w-\hat\mu\). Differentiating again:

\[
\frac{\partial^2 f}{\partial w_k\partial w_l}=\lambda\Sigma_{kl};\qquad H=\lambda\Sigma.
\]

A covariance matrix is positive semidefinite, so for \(\lambda>0\) the Hessian is PSD and the quadratic objective is convex. Nonzero diagonal regularization makes the numerical covariance strictly positive definite in nondegenerate settings.

## 3. Finite-difference verification

For basis vector \(e_i\), use centered differences:

\[
\partial_i f(w)\approx\frac{f(w+\epsilon e_i)-f(w-\epsilon e_i)}{2\epsilon},\quad
H_{:i}(w)\approx\frac{\nabla f(w+\epsilon e_i)-\nabla f(w-\epsilon e_i)}{2\epsilon}.
\]

Run `python -m examples.inspect_derivatives` or `python -m pytest -q`. These tests run without market data or internet access.

## 4. Constraint logic and KKT interpretation

With equality multiplier \(\nu\), non-negativity multipliers \(\alpha_i\ge0\), and upper-bound multipliers \(\beta_i\ge0\):

\[
\mathcal{L}=f(w)+\nu(\mathbf1^\top w-1)-\alpha^\top w+\beta^\top(w-c\mathbf1).
\]

Stationarity is \(\lambda\Sigma w-\hat\mu+\nu\mathbf1-\alpha+\beta=0\), with complementarity \(\alpha_i w_i=0\), \(\beta_i(w_i-c)=0\). Constraints plus PSD Hessian justify convex optimization; KKT conditions are necessary and sufficient under suitable constraint qualification. The code uses SLSQP with an analytic Jacobian and checks feasibility after solution. It does *not* claim to compute explicit KKT multipliers.

## 5. Risk contribution as an additional derivative application

Portfolio variance: \(V(w)=w^\top\Sigma w\), so \(\nabla V=2\Sigma w\).
Portfolio volatility: \(\sigma_p(w)=\sqrt{w^\top\Sigma w}\), so \(\nabla\sigma_p=\Sigma w/\sigma_p\) when \(\sigma_p>0\).

By Euler's theorem for homogeneous functions:

\[
\mathrm{RC}_i=w_i\frac{\partial\sigma_p}{\partial w_i}=\frac{w_i(\Sigma w)_i}{\sigma_p},\quad \sum_i\mathrm{RC}_i=\sigma_p.
\]

This decomposition is implemented in `diagnostics.py`; variance contributions in `optimization.py` sum to total variance instead.

## 6. Consistent forecasting units

The Random Forest predicts *simple cumulative return over the next 21 trading days*. Covariance is estimated from the previous 63 daily simple returns and scaled by 21. This commonly used approximation assumes serial dependence between daily returns is negligible. The model does not explicitly forecast 21-day covariance from a serial-correlated process, and does not promise calibrated forecasts.

## What this does not prove

This work showcases multivariate differentiation, gradients, Hessians, convex quadratic programs, and mathematical finance. It does not cover double integrals, triple integrals, Jacobian change of variables in multiple integration, vector calculus theorems, or all material usually examined in a dedicated multivariable calculus course.
