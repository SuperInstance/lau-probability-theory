use crate::distributions::{Distribution, SampleStats};

/// Verify the Central Limit Theorem: sample means of any distribution converge to Normal.
pub fn verify_clt<D: Distribution>(dist: &D, sample_size: usize, num_means: usize, rng: &mut impl rand::Rng) -> SampleStats {
    let mut means: Vec<f64> = Vec::with_capacity(num_means);
    for _ in 0..num_means {
        let sample: Vec<f64> = (0..sample_size).map(|_| dist.sample(rng)).collect();
        let mean = sample.iter().sum::<f64>() / sample_size as f64;
        means.push(mean);
    }
    SampleStats::from_samples(&means)
}

/// Verify the Law of Large Numbers: sample mean converges to true mean.
pub fn verify_lln<D: Distribution>(dist: &D, max_samples: usize, rng: &mut impl rand::Rng) -> Vec<(usize, f64)> {
    let true_mean = dist.mean();
    let mut running_sum = 0.0;
    let mut results = Vec::new();
    for i in 1..=max_samples {
        running_sum += dist.sample(rng);
        let sample_mean = running_sum / i as f64;
        results.push((i, (sample_mean - true_mean).abs()));
    }
    results
}

/// Compute empirical CDF from samples.
pub fn empirical_cdf(samples: &[f64], x: f64) -> f64 {
    let count = samples.iter().filter(|&&s| s <= x).count();
    count as f64 / samples.len() as f64
}

/// Kolmogorov-Smirnov statistic: max |empirical CDF - theoretical CDF|.
pub fn ks_statistic<D: Distribution>(dist: &D, samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let mut max_diff = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let emp_cdf = (i + 1) as f64 / n;
        let theo_cdf = dist.cdf(x);
        let diff = (emp_cdf - theo_cdf).abs();
        if diff > max_diff { max_diff = diff; }
    }
    max_diff
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete::{Bernoulli, Binomial};
    use crate::continuous::Exponential;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_clt_convergence_bernoulli() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Bernoulli::new(0.3);
        let stats = verify_clt(&dist, 100, 5000, &mut rng);
        // Sample means should be close to population mean
        let expected_mean = 0.3;
        assert!((stats.mean - expected_mean).abs() < 0.02);
        // Variance of sample means should be sigma^2/n = 0.21/100 = 0.0021
        let expected_var = 0.21 / 100.0;
        assert!((stats.variance - expected_var).abs() < 0.001);
    }

    #[test]
    fn test_clt_convergence_exponential() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Exponential::new(2.0);
        let stats = verify_clt(&dist, 50, 5000, &mut rng);
        assert!((stats.mean - 0.5).abs() < 0.02);
        // Var of mean = Var(X)/n = 0.25/50 = 0.005
        assert!((stats.variance - 0.005).abs() < 0.002);
    }

    #[test]
    fn test_lln_convergence() {
        let mut rng = StdRng::seed_from_u64(42);
        let dist = Binomial::new(10, 0.5);
        let results = verify_lln(&dist, 10000, &mut rng);
        // Error should decrease
        let early_error = results[99].1;
        let late_error = results[9999].1;
        // Late error should generally be smaller
        assert!(late_error < early_error * 5.0); // Allow some slack
        // Final error should be small
        assert!(late_error < 0.1);
    }
}
