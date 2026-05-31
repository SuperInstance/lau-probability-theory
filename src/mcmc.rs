use rand::Rng;
use serde::{Deserialize, Serialize};

/// Metropolis-Hastings sampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetropolisHastings {
    /// Proposal standard deviation
    pub proposal_sigma: f64,
    /// Number of burn-in steps
    pub burn_in: usize,
    /// Thinning interval
    pub thin: usize,
}

impl MetropolisHastings {
    pub fn new(proposal_sigma: f64, burn_in: usize, thin: usize) -> Self {
        MetropolisHastings { proposal_sigma, burn_in, thin }
    }

    /// Run Metropolis-Hastings for a univariate target distribution.
    /// `log_target` is the log of the target density (up to a constant).
    /// `initial` is the starting value.
    /// `n_samples` is the number of samples to collect (after burn-in and thinning).
    pub fn sample<F: Fn(f64) -> f64>(
        &self,
        log_target: F,
        initial: f64,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> MCMCResult {
        let mut current = initial;
        let mut current_log_p = log_target(current);
        let mut samples = Vec::with_capacity(n_samples);
        let mut total_steps = 0usize;
        let mut accepted = 0usize;

        let total_needed = self.burn_in + n_samples * self.thin;

        for step in 0..=total_needed {
            // Propose
            let proposal = current + self.proposal_sigma * rand_distr::StandardNormal.sample(rng);
            let proposal_log_p = log_target(proposal);

            // Accept/reject
            let log_alpha = proposal_log_p - current_log_p;
            if log_alpha.is_nan() {
                // Reject NaN proposals
            } else if log_alpha >= 0.0 || rng.gen::<f64>().ln() < log_alpha {
                current = proposal;
                current_log_p = proposal_log_p;
                accepted += 1;
            }

            total_steps += 1;

            // Collect after burn-in, with thinning
            if step > self.burn_in && (step - self.burn_in) % self.thin == 0 {
                samples.push(current);
                if samples.len() >= n_samples { break; }
            }
        }

        let acceptance_rate = accepted as f64 / total_steps as f64;
        MCMCResult {
            samples,
            acceptance_rate,
            total_steps,
        }
    }
}

/// Result of an MCMC run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCMCResult {
    pub samples: Vec<f64>,
    pub acceptance_rate: f64,
    pub total_steps: usize,
}

impl MCMCResult {
    pub fn mean(&self) -> f64 {
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let n = self.samples.len();
        self.samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64
    }
    pub fn quantile(&self, p: f64) -> f64 {
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (sorted.len() - 1) as f64) as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
}

/// Simple rand_distr wrapper for standard normal sampling.
mod rand_distr {
    use rand::Rng;
    pub struct StandardNormal;
    impl StandardNormal {
        pub fn sample(&self, rng: &mut impl Rng) -> f64 {
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_mh_samples_from_normal() {
        let mut rng = StdRng::seed_from_u64(42);
        // Target: Normal(0, 1)
        let log_target = |x: f64| -0.5 * x * x;
        let mh = MetropolisHastings::new(1.0, 1000, 5);
        let result = mh.sample(log_target, 0.0, 5000, &mut rng);

        // Acceptance rate should be reasonable (20-60%)
        assert!(result.acceptance_rate > 0.2 && result.acceptance_rate < 0.8);

        // Sample mean should be close to 0
        assert!(result.mean().abs() < 0.1);

        // Sample variance should be close to 1
        assert!((result.variance() - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_mh_samples_from_exponential() {
        let mut rng = StdRng::seed_from_u64(42);
        // Target: Exponential(1) - log density: -x for x > 0
        let log_target = |x: f64| if x > 0.0 { -x } else { f64::NEG_INFINITY };
        let mh = MetropolisHastings::new(1.0, 1000, 5);
        let result = mh.sample(log_target, 1.0, 5000, &mut rng);

        // Mean should be close to 1
        assert!((result.mean() - 1.0).abs() < 0.2);
    }
}
