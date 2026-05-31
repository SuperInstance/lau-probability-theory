use serde::{Deserialize, Serialize};
use crate::continuous::standard_normal_cdf;

/// Result of a hypothesis test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub reject_null: bool,
    pub confidence_level: f64,
}

/// One-sample z-test (known variance).
pub fn z_test(sample_mean: f64, mu_0: f64, sigma: f64, n: usize, alpha: f64) -> TestResult {
    let se = sigma / (n as f64).sqrt();
    let z = (sample_mean - mu_0) / se;
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
    TestResult {
        statistic: z,
        p_value,
        reject_null: p_value < alpha,
        confidence_level: 1.0 - alpha,
    }
}

/// One-sample t-test (unknown variance).
pub fn t_test(sample_mean: f64, sample_std: f64, mu_0: f64, n: usize, alpha: f64) -> TestResult {
    let se = sample_std / (n as f64).sqrt();
    let t = (sample_mean - mu_0) / se;
    let df = (n - 1) as f64;
    let p_value = 2.0 * (1.0 - t_cdf(t.abs(), df));
    TestResult {
        statistic: t,
        p_value,
        reject_null: p_value < alpha,
        confidence_level: 1.0 - alpha,
    }
}

/// Chi-squared goodness-of-fit test.
pub fn chi_squared_test(observed: &[f64], expected: &[f64], alpha: f64) -> TestResult {
    assert_eq!(observed.len(), expected.len());
    let chi2: f64 = observed.iter().zip(expected.iter())
        .map(|(o, e)| (o - e).powi(2) / e)
        .sum();
    let df = observed.len() as f64 - 1.0;
    let p_value = 1.0 - chi2_cdf(chi2, df);
    TestResult {
        statistic: chi2,
        p_value,
        reject_null: p_value < alpha,
        confidence_level: 1.0 - alpha,
    }
}

/// Chi-squared CDF using regularized gamma function.
/// P(chi2; k) = regularized_gamma_p(k/2, x/2)
pub fn chi2_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    crate::continuous::regularized_gamma_p(df / 2.0, x / 2.0)
}


/// Approximate t-distribution CDF using normal approximation for large df.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    // Use the approximation: t_cdf ≈ normal_cdf for large df
    // For small df, use a more careful approximation
    if df > 30.0 {
        return standard_normal_cdf(t);
    }
    // Hill's approximation
    let x = df / (df + t * t);
    let p = 0.5 * regularized_beta(x, df / 2.0, 0.5);
    if t >= 0.0 { 1.0 - p } else { p }
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    crate::continuous::regularized_beta(x, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_test_reject() {
        // Sample mean 105, H0: mu=100, sigma=15, n=100
        let result = z_test(105.0, 100.0, 15.0, 100, 0.05);
        assert!((result.statistic - (5.0 / 1.5)).abs() < 1e-10);
        assert!(result.p_value < 0.05);
        assert!(result.reject_null);
    }

    #[test]
    fn test_z_test_fail_to_reject() {
        let result = z_test(101.0, 100.0, 15.0, 25, 0.05);
        assert!(result.p_value > 0.05);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_t_test_basic() {
        // Large sample, should behave like z-test
        let result = t_test(105.0, 15.0, 100.0, 100, 0.05);
        assert!(result.p_value < 0.1);
    }

    #[test]
    fn test_chi_squared_uniform() {
        // Observed counts close to expected - should not reject
        let observed = vec![48.0, 52.0, 50.0, 51.0, 49.0, 50.0];
        let expected = vec![50.0; 6];
        let result = chi_squared_test(&observed, &expected, 0.05);
        assert!(result.p_value > 0.05);
        assert!(!result.reject_null);
    }

    #[test]
    fn test_chi_squared_non_uniform() {
        // Observed counts very different from expected
        let observed = vec![80.0, 10.0, 10.0, 10.0, 10.0, 80.0];
        let expected = vec![50.0; 6];
        let result = chi_squared_test(&observed, &expected, 0.05);
        assert!(result.p_value < 0.05);
        assert!(result.reject_null);
    }

    #[test]
    fn test_chi2_cdf_known_value() {
        // chi2_cdf(3.841, 1) should be approximately 0.95
        let cdf_val = chi2_cdf(3.841, 1.0);
        assert!((cdf_val - 0.95).abs() < 0.05, "got {}", cdf_val);
    }
}
