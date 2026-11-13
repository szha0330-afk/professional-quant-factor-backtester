"""Standalone calculus check: run `python -m examples.inspect_derivatives`."""
import numpy as np
from quant_research.optimization import objective, gradient, hessian
from quant_research.diagnostics import derivative_diagnostics, euler_volatility_contributions

w = np.array([0.3, 0.4, 0.3])
mu = np.array([0.02, 0.015, 0.025])
sigma = np.array([[0.04, 0.006, 0.004],
                  [0.006, 0.03, 0.008],
                  [0.004, 0.008, 0.05]])
lam = 8.0
print("Portfolio objective:", round(objective(w, mu, sigma, lam), 8))
print("Gradient:", gradient(w, mu, sigma, lam))
print("Hessian:\n", hessian(sigma, lam))
print("Derivative diagnostics:", derivative_diagnostics(w, mu, sigma, lam))
print("Euler volatility contributions:", euler_volatility_contributions(w, sigma))
