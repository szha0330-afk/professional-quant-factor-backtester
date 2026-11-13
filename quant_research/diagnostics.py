"""Finite-difference diagnostics and Euler risk contribution decomposition."""
from __future__ import annotations
import numpy as np
from .optimization import objective, gradient, hessian


def finite_difference_gradient(w, mu, sigma, risk_aversion, eps: float = 1e-6):
    w = np.asarray(w, float)
    eye = np.eye(w.size)
    return np.array([(objective(w + eps * e, mu, sigma, risk_aversion) -
                      objective(w - eps * e, mu, sigma, risk_aversion)) / (2 * eps)
                     for e in eye])


def finite_difference_hessian(w, mu, sigma, risk_aversion, eps: float = 1e-5):
    w = np.asarray(w, float)
    eye = np.eye(w.size)
    return np.column_stack([(gradient(w + eps * e, mu, sigma, risk_aversion) -
                             gradient(w - eps * e, mu, sigma, risk_aversion)) / (2 * eps)
                            for e in eye])


def derivative_diagnostics(w, mu, sigma, risk_aversion=8.0) -> dict[str, float]:
    analytic_grad = gradient(w, mu, sigma, risk_aversion)
    analytic_hess = hessian(sigma, risk_aversion)
    numeric_grad = finite_difference_gradient(w, mu, sigma, risk_aversion)
    numeric_hess = finite_difference_hessian(w, mu, sigma, risk_aversion)
    return {
        "max_gradient_abs_error": float(np.max(np.abs(analytic_grad - numeric_grad))),
        "max_hessian_abs_error": float(np.max(np.abs(analytic_hess - numeric_hess))),
        "hessian_min_eigenvalue": float(np.linalg.eigvalsh(analytic_hess).min()),
    }


def euler_volatility_contributions(weights, sigma) -> np.ndarray:
    """Contributions sum to portfolio standard deviation, for sigma PSD."""
    w = np.asarray(weights, float)
    sigma = np.asarray(sigma, float)
    variance = float(w @ sigma @ w)
    if variance <= 0:
        raise ValueError("Portfolio variance must be positive")
    return w * (sigma @ w) / np.sqrt(variance)
