use serde::{Deserialize, Serialize};
use crate::bayesian::{BetaPosterior, normal_normal_update, normal_credible_interval};

/// Represents an agent's belief about a binary outcome (e.g., "will action succeed?").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryBelief {
    pub posterior: BetaPosterior,
    pub label: String,
}

impl BinaryBelief {
    pub fn new(label: &str, alpha: f64, beta: f64) -> Self {
        BinaryBelief {
            posterior: BetaPosterior::new(alpha, beta),
            label: label.to_string(),
        }
    }

    /// Uninformative prior (Uniform).
    pub fn uninformative(label: &str) -> Self {
        Self::new(label, 1.0, 1.0)
    }

    /// Update belief with new evidence.
    pub fn observe(&mut self, success: bool) {
        let (s, f) = if success { (1, 0) } else { (0, 1) };
        self.posterior = self.posterior.update(s, f);
    }

    /// Expected probability of success.
    pub fn expected_probability(&self) -> f64 {
        self.posterior.mean()
    }

    /// Uncertainty (standard deviation) of the belief.
    pub fn uncertainty(&self) -> f64 {
        self.posterior.variance().sqrt()
    }

    /// 95% credible interval for the probability.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        self.posterior.credible_interval(0.95)
    }

    /// Confidence level: 1 - uncertainty (normalized).
    pub fn confidence(&self) -> f64 {
        1.0 - self.uncertainty() * 4.0 // Scale so variance 0.0625 → confidence 0.5
    }
}

/// Represents an agent's belief about a continuous quantity (e.g., "how long will this take?").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousBelief {
    pub mu: f64,
    pub sigma: f64,
    pub label: String,
    pub observations: usize,
}

impl ContinuousBelief {
    pub fn new(label: &str, mu: f64, sigma: f64) -> Self {
        ContinuousBelief { mu, sigma, label: label.to_string(), observations: 0 }
    }

    /// Update with new observation (known measurement noise sigma_likelihood).
    pub fn observe(&mut self, value: f64, sigma_likelihood: f64) {
        let data = vec![value];
        let post = normal_normal_update(self.mu, self.sigma, sigma_likelihood, &data);
        self.mu = post.mu_post;
        self.sigma = post.sigma_post;
        self.observations += 1;
    }

    /// Expected value.
    pub fn expected_value(&self) -> f64 { self.mu }

    /// Uncertainty.
    pub fn uncertainty(&self) -> f64 { self.sigma }

    /// 95% credible interval.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        normal_credible_interval(self.mu, self.sigma, 0.95)
    }
}

/// A simple decision model under uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub options: Vec<DecisionOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOption {
    pub label: String,
    pub expected_utility: f64,
    pub utility_std: f64,
}

impl Decision {
    pub fn new() -> Self { Decision { options: Vec::new() } }

    pub fn add_option(&mut self, label: &str, expected_utility: f64, utility_std: f64) {
        self.options.push(DecisionOption {
            label: label.to_string(),
            expected_utility,
            utility_std,
        });
    }

    /// Choose the option with highest expected utility.
    pub fn best_option(&self) -> &DecisionOption {
        self.options.iter().max_by(|a, b| a.expected_utility.partial_cmp(&b.expected_utility).unwrap()).unwrap()
    }

    /// Probability that option i has higher utility than option j (normal approximation).
    pub fn prob_i_beats_j(&self, i: usize, j: usize) -> f64 {
        let oi = &self.options[i];
        let oj = &self.options[j];
        let diff_mean = oi.expected_utility - oj.expected_utility;
        let diff_std = (oi.utility_std.powi(2) + oj.utility_std.powi(2)).sqrt();
        if diff_std < 1e-10 { return if diff_mean > 0.0 { 1.0 } else { 0.0 }; }
        crate::continuous::standard_normal_cdf(diff_mean / diff_std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_belief_update() {
        let mut belief = BinaryBelief::uninformative("task_success");
        assert!((belief.expected_probability() - 0.5).abs() < 0.01);

        belief.observe(true);
        belief.observe(true);
        belief.observe(true);
        belief.observe(false);

        // After 3 successes, 1 failure: Beta(4, 2), mean = 4/6 ≈ 0.667
        assert!((belief.expected_probability() - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_binary_belief_confidence_increases() {
        let mut belief = BinaryBelief::uninformative("test");
        let initial_uncertainty = belief.uncertainty();

        for _ in 0..100 {
            belief.observe(true);
        }
        assert!(belief.uncertainty() < initial_uncertainty);
    }

    #[test]
    fn test_continuous_belief_update() {
        let mut belief = ContinuousBelief::new("duration", 10.0, 5.0);
        belief.observe(8.0, 2.0);
        // Posterior mean should be between 10 and 8
        assert!(belief.mu > 8.0 && belief.mu < 10.0);
        // Uncertainty should decrease
        assert!(belief.sigma < 5.0);
    }

    #[test]
    fn test_decision_best_option() {
        let mut decision = Decision::new();
        decision.add_option("A", 5.0, 1.0);
        decision.add_option("B", 8.0, 2.0);
        decision.add_option("C", 3.0, 0.5);
        assert_eq!(decision.best_option().label, "B");
    }

    #[test]
    fn test_decision_prob_comparison() {
        let mut decision = Decision::new();
        decision.add_option("A", 5.0, 1.0);
        decision.add_option("B", 5.0, 1.0);
        // Same expected utility → 50/50
        let prob = decision.prob_i_beats_j(0, 1);
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_binary_belief_credible_interval() {
        let mut belief = BinaryBelief::uninformative("test");
        for _ in 0..50 {
            belief.observe(true);
        }
        let (lower, upper) = belief.credible_interval_95();
        // After 50 successes, should be very confident near 1
        assert!(lower > 0.85);
        assert!(upper <= 1.0);
    }

    #[test]
    fn test_agent_belief_scenario() {
        // Agent deciding between two tasks with uncertain rewards
        let mut task_a = BinaryBelief::uninformative("task_a_success");
        let mut task_b = BinaryBelief::uninformative("task_b_success");

        // Task A: 8 successes out of 10
        for _ in 0..8 { task_a.observe(true); }
        for _ in 0..2 { task_a.observe(false); }

        // Task B: 3 successes out of 10
        for _ in 0..3 { task_b.observe(true); }
        for _ in 0..7 { task_b.observe(false); }

        // Task A should have higher expected success
        assert!(task_a.expected_probability() > task_b.expected_probability());
    }
}
