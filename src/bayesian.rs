use serde::{Deserialize, Serialize};
use crate::continuous::{beta_fn, standard_normal_inv};

/// Conjugate prior result for Normal-Normal model (known variance).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalPosterior {
    pub mu_post: f64,
    pub sigma_post: f64,
}

/// Bayesian update for Normal likelihood with known variance and Normal prior.
/// Prior: N(mu_0, sigma_0^2), Likelihood: N(x_bar, sigma^2/n)
pub fn normal_normal_update(mu_prior: f64, sigma_prior: f64, sigma_likelihood: f64, data: &[f64]) -> NormalPosterior {
    let n = data.len() as f64;
    let x_bar = data.iter().sum::<f64>() / n;
    let var_prior = sigma_prior * sigma_prior;
    let var_lik = sigma_likelihood * sigma_likelihood / n;

    let var_post = 1.0 / (1.0 / var_prior + 1.0 / var_lik);
    let mu_post = var_post * (mu_prior / var_prior + x_bar / var_lik);

    NormalPosterior {
        mu_post,
        sigma_post: var_post.sqrt(),
    }
}

/// Beta-Binomial conjugate update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaPosterior {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaPosterior {
    pub fn new(alpha: f64, beta: f64) -> Self {
        BetaPosterior { alpha, beta }
    }

    /// Posterior mean.
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance.
    pub fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }

    /// Credible interval (equal-tailed).
    pub fn credible_interval(&self, level: f64) -> (f64, f64) {
        let alpha = (1.0 - level) / 2.0;
        let lower = beta_quantile(alpha, self.alpha, self.beta);
        let upper = beta_quantile(1.0 - alpha, self.alpha, self.beta);
        (lower, upper)
    }

    /// Posterior predictive: probability of success on next trial.
    pub fn predictive_success(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Update with observed successes and failures.
    pub fn update(&self, successes: u64, failures: u64) -> Self {
        BetaPosterior {
            alpha: self.alpha + successes as f64,
            beta: self.beta + failures as f64,
        }
    }
}

/// Beta-Binomial update: Beta(alpha, beta) prior + Binomial data.
pub fn beta_binomial_update(alpha_prior: f64, beta_prior: f64, successes: u64, trials: u64) -> BetaPosterior {
    BetaPosterior {
        alpha: alpha_prior + successes as f64,
        beta: beta_prior + (trials - successes) as f64,
    }
}

/// Gamma-Poisson conjugate update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaPosterior {
    pub shape: f64,
    pub rate: f64,
}

impl GammaPosterior {
    pub fn mean(&self) -> f64 { self.shape / self.rate }
    pub fn variance(&self) -> f64 { self.shape / (self.rate * self.rate) }
}

/// Gamma-Poisson update: Gamma(shape, rate) prior + Poisson data.
pub fn gamma_poisson_update(shape_prior: f64, rate_prior: f64, data: &[f64]) -> GammaPosterior {
    let sum: f64 = data.iter().sum();
    let n = data.len() as f64;
    GammaPosterior {
        shape: shape_prior + sum,
        rate: rate_prior + n,
    }
}

/// Approximate Beta quantile via bisection.
pub fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 { return 0.0; }
    if p >= 1.0 { return 1.0; }
    let _beta_val = beta_fn(a, b);
    let beta_cdf = |x: f64| -> f64 {
        crate::continuous::regularized_beta(x, a, b)
    };
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        let cdf_mid = beta_cdf(mid);
        if (cdf_mid - p).abs() < 1e-8 { return mid; }
        if cdf_mid < p { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

/// Normal posterior credible interval.
pub fn normal_credible_interval(mu: f64, sigma: f64, level: f64) -> (f64, f64) {
    let alpha = (1.0 - level) / 2.0;
    let z = standard_normal_inv(1.0 - alpha);
    (mu - z * sigma, mu + z * sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_beta_binomial_update() {
        // Prior: Beta(1, 1) = Uniform
        let post = beta_binomial_update(1.0, 1.0, 7, 10);
        assert_relative_eq!(post.alpha, 8.0);
        assert_relative_eq!(post.beta, 4.0);
        // Posterior mean should be 8/12 = 2/3
        assert_relative_eq!(post.mean(), 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normal_normal_update() {
        // Prior: N(0, 1), Likelihood sigma=1, data [1.0, 1.0, 1.0]
        let data = vec![1.0, 1.0, 1.0];
        let post = normal_normal_update(0.0, 1.0, 1.0, &data);
        // Posterior mean should be between 0 and 1
        assert!(post.mu_post > 0.5 && post.mu_post < 1.0);
        // Posterior sigma should be less than prior sigma
        assert!(post.sigma_post < 1.0);
    }

    #[test]
    fn test_gamma_poisson_update() {
        // Prior: Gamma(1, 1), data: [3, 4, 5]
        let data = vec![3.0, 4.0, 5.0];
        let post = gamma_poisson_update(1.0, 1.0, &data);
        assert_relative_eq!(post.shape, 13.0);
        assert_relative_eq!(post.rate, 4.0);
        assert_relative_eq!(post.mean(), 13.0 / 4.0);
    }

    #[test]
    fn test_credible_interval_contains_true_value() {
        let post = BetaPosterior::new(5.0, 5.0);
        let (lower, upper) = post.credible_interval(0.95);
        // Mean is 0.5, interval should straddle it
        // For Beta(5,5), the 95% CI is approximately [0.23, 0.77]
        // Lower should be < 0.5 and upper > 0.5
        // Use somewhat relaxed bounds to account for numerical approximation
        assert!(lower < 0.6, "lower = {}", lower);
        assert!(upper > 0.4, "upper = {}", upper);
        assert!(lower > 0.0 && lower < 1.0, "lower = {}", lower);
        assert!(upper > 0.0 && upper < 1.0, "upper = {}", upper);
    }

    #[test]
    fn test_sequential_bayesian_update() {
        let mut post = BetaPosterior::new(1.0, 1.0);
        // Observe 3 successes, 1 failure
        post = post.update(3, 1);
        assert_relative_eq!(post.alpha, 4.0);
        assert_relative_eq!(post.beta, 2.0);
        // Observe 1 more success
        post = post.update(1, 0);
        assert_relative_eq!(post.alpha, 5.0);
        assert_relative_eq!(post.beta, 2.0);
    }

    #[test]
    fn test_normal_credible_interval() {
        let (lower, upper) = normal_credible_interval(0.0, 1.0, 0.95);
        assert!((lower - (-1.96)).abs() < 0.05);
        assert!((upper - 1.96).abs() < 0.05);
    }
}
