"""Bayesian probability update helpers for the quant bot."""


def bayes_update(
    prior: float, likelihood_true: float, likelihood_false: float
) -> float:
    """Update a prior probability using Bayes' rule."""
    safe_prior = min(max(prior, 1e-9), 1 - 1e-9)
    numerator = likelihood_true * safe_prior
    denominator = numerator + (likelihood_false * (1 - safe_prior))

    if denominator == 0:
        return safe_prior

    posterior = numerator / denominator
    return min(max(posterior, 0.0), 1.0)


if __name__ == "__main__":
    prior = 0.50
    likelihood_true = 0.70
    likelihood_false = 0.35
    posterior = bayes_update(prior, likelihood_true, likelihood_false)
    print("Posterior:", round(posterior, 4))
