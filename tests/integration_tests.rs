use lau_probability_theory::distributions::Distribution;
use lau_probability_theory::discrete::*;
use lau_probability_theory::continuous::*;
use lau_probability_theory::mcmc::*;
use lau_probability_theory::application::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use approx::assert_relative_eq;

fn rng() -> StdRng { StdRng::seed_from_u64(42) }

// ===== Discrete Distributions =====

#[test]
fn test_bernoulli_mean_variance() {
    let b = Bernoulli::new(0.3);
    assert_relative_eq!(b.mean(), 0.3);
    assert_relative_eq!(b.variance(), 0.21);
}

#[test]
fn test_bernoulli_cdf() {
    let b = Bernoulli::new(0.3);
    assert_relative_eq!(b.cdf(-0.5), 0.0);
    assert_relative_eq!(b.cdf(0.0), 0.7);
    assert_relative_eq!(b.cdf(0.5), 0.7);
    assert_relative_eq!(b.cdf(1.0), 1.0);
}

#[test]
fn test_bernoulli_mgf() {
    let b = Bernoulli::new(0.5);
    assert_relative_eq!(b.mgf(0.0), 1.0);
}

#[test]
fn test_binomial_mean_variance() {
    let b = Binomial::new(20, 0.4);
    assert_relative_eq!(b.mean(), 8.0);
    assert_relative_eq!(b.variance(), 4.8);
}

#[test]
fn test_binomial_cdf_bounds() {
    let b = Binomial::new(5, 0.5);
    assert!(b.cdf(-1.0) < 0.05);
    assert!(b.cdf(5.0) > 0.99);
}

#[test]
fn test_binomial_mgf() {
    let b = Binomial::new(10, 0.3);
    assert_relative_eq!(b.mgf(0.0), 1.0);
}

#[test]
fn test_poisson_mean_variance() {
    let p = Poisson::new(5.0);
    assert_relative_eq!(p.mean(), 5.0);
    assert_relative_eq!(p.variance(), 5.0);
}

#[test]
fn test_poisson_mgf() {
    let p = Poisson::new(3.0);
    assert_relative_eq!(p.mgf(0.0), 1.0);
}

#[test]
fn test_geometric_mean_variance() {
    let g = Geometric::new(0.2);
    assert_relative_eq!(g.mean(), 5.0);
    assert_relative_eq!(g.variance(), 20.0);
}

#[test]
fn test_geometric_cdf() {
    let g = Geometric::new(0.5);
    assert_relative_eq!(g.cdf(0.0), 0.0);
    assert!(g.cdf(1.0) > 0.0);
    assert!(g.cdf(10.0) > 0.99);
}

#[test]
fn test_hypergeometric_mean_variance() {
    let h = Hypergeometric::new(100, 30, 20);
    let expected_mean = 20.0 * 30.0 / 100.0;
    assert_relative_eq!(h.mean(), expected_mean);
    assert!(h.variance() > 0.0);
}

#[test]
fn test_hypergeometric_cdf() {
    let h = Hypergeometric::new(10, 5, 3);
    assert!(h.cdf(-1.0) < 0.1);
    assert!(h.cdf(3.0) > 0.99);
}

// ===== Continuous Distributions =====

#[test]
fn test_uniform_mean_variance() {
    let u = Uniform::new(0.0, 1.0);
    assert_relative_eq!(u.mean(), 0.5);
    assert_relative_eq!(u.variance(), 1.0 / 12.0);
}

#[test]
fn test_uniform_cdf() {
    let u = Uniform::new(2.0, 5.0);
    assert_relative_eq!(u.cdf(2.0), 0.0);
    assert_relative_eq!(u.cdf(3.5), 0.5);
    assert_relative_eq!(u.cdf(5.0), 1.0);
}

#[test]
fn test_normal_mean_variance() {
    let n = Normal::new(5.0, 2.0);
    assert_relative_eq!(n.mean(), 5.0);
    assert_relative_eq!(n.variance(), 4.0);
}

#[test]
fn test_normal_pdf_symmetric() {
    let n = Normal::new(0.0, 1.0);
    assert_relative_eq!(n.pdf(-1.0), n.pdf(1.0));
}

#[test]
fn test_normal_cdf_symmetric() {
    let n = Normal::standard();
    assert!((n.cdf(0.0) - 0.5).abs() < 1e-6);
    assert!((n.cdf(1.96) - (1.0 - n.cdf(-1.96))).abs() < 1e-6);
}

#[test]
fn test_normal_mgf() {
    let n = Normal::new(0.0, 1.0);
    assert_relative_eq!(n.mgf(0.0), 1.0);
}

#[test]
fn test_exponential_mean_variance() {
    let e = Exponential::new(2.0);
    assert_relative_eq!(e.mean(), 0.5);
    assert_relative_eq!(e.variance(), 0.25);
}

#[test]
fn test_exponential_cdf() {
    let e = Exponential::new(1.0);
    assert_relative_eq!(e.cdf(0.0), 0.0);
    let at_1 = e.cdf(1.0);
    assert!((at_1 - 0.6321).abs() < 0.01);
}

#[test]
fn test_gamma_mean_variance() {
    let g = Gamma::new(2.0, 3.0);
    assert_relative_eq!(g.mean(), 2.0 / 3.0);
    assert_relative_eq!(g.variance(), 2.0 / 9.0);
}

#[test]
fn test_beta_mean_variance() {
    let b = Beta::new(2.0, 5.0);
    assert_relative_eq!(b.mean(), 2.0 / 7.0);
    let expected_var = 2.0 * 5.0 / (49.0 * 8.0);
    assert_relative_eq!(b.variance(), expected_var);
}

#[test]
fn test_beta_pdf_bounds() {
    let b = Beta::new(2.0, 3.0);
    assert_relative_eq!(b.pdf(0.0), 0.0);
    assert_relative_eq!(b.pdf(1.0), 0.0);
    assert!(b.pdf(0.5) > 0.0);
}

// ===== Sampling =====

#[test]
fn test_bernoulli_sampling() {
    let b = Bernoulli::new(0.7);
    let mut rng = rng();
    let samples: Vec<f64> = (0..1000).map(|_| b.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / 1000.0;
    assert!((mean - 0.7).abs() < 0.05);
}

#[test]
fn test_normal_sampling() {
    let n = Normal::new(10.0, 2.0);
    let mut rng = rng();
    let samples: Vec<f64> = (0..5000).map(|_| n.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / 5000.0;
    assert!((mean - 10.0).abs() < 0.1);
}

#[test]
fn test_exponential_sampling() {
    let e = Exponential::new(1.0);
    let mut rng = rng();
    let samples: Vec<f64> = (0..5000).map(|_| e.sample(&mut rng)).collect();
    let mean = samples.iter().sum::<f64>() / 5000.0;
    assert!((mean - 1.0).abs() < 0.1);
}

// ===== MGF properties =====

#[test]
fn test_mgf_at_zero_is_one() {
    // For any distribution, MGF(0) = 1
    assert_relative_eq!(Bernoulli::new(0.5).mgf(0.0), 1.0);
    assert_relative_eq!(Binomial::new(10, 0.3).mgf(0.0), 1.0);
    assert_relative_eq!(Poisson::new(5.0).mgf(0.0), 1.0);
    assert_relative_eq!(Uniform::new(0.0, 1.0).mgf(0.0), 1.0);
    assert_relative_eq!(Normal::new(0.0, 1.0).mgf(0.0), 1.0);
    assert_relative_eq!(Exponential::new(1.0).mgf(0.0), 1.0);
    assert_relative_eq!(Gamma::new(2.0, 1.0).mgf(0.0), 1.0);
}

// ===== Helper functions =====

#[test]
fn test_standard_normal_cdf() {
    assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
    assert!(standard_normal_cdf(1.96) > 0.97);
    assert!(standard_normal_cdf(-1.96) < 0.03);
}

#[test]
fn test_standard_normal_inv_roundtrip() {
    for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let x = standard_normal_inv(p);
        let p_back = standard_normal_cdf(x);
        assert!((p_back - p).abs() < 0.01, "p={}, got={}", p, p_back);
    }
}

#[test]
fn test_gamma_fn_values() {
    assert_relative_eq!(gamma_fn(1.0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(gamma_fn(2.0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(gamma_fn(3.0), 2.0, epsilon = 1e-10);
    assert_relative_eq!(gamma_fn(5.0), 24.0, epsilon = 1e-8);
}

#[test]
fn test_beta_fn_values() {
    // B(1,1) = 1
    assert_relative_eq!(beta_fn(1.0, 1.0), 1.0, epsilon = 1e-10);
    // B(2,2) = 1/6
    assert_relative_eq!(beta_fn(2.0, 2.0), 1.0 / 6.0, epsilon = 1e-10);
}

// ===== MCMC =====

#[test]
fn test_mh_acceptance_rate_reasonable() {
    let mut rng = rng();
    let log_target = |x: f64| -0.5 * x * x;
    let mh = MetropolisHastings::new(0.5, 500, 3);
    let result = mh.sample(log_target, 0.0, 2000, &mut rng);
    assert!(result.acceptance_rate > 0.3 && result.acceptance_rate < 0.9);
}

#[test]
fn test_mh_quantiles() {
    let mut rng = rng();
    let log_target = |x: f64| -0.5 * x * x;
    let mh = MetropolisHastings::new(1.0, 1000, 5);
    let result = mh.sample(log_target, 0.0, 5000, &mut rng);
    // For standard normal, 2.5th percentile ≈ -1.96, 97.5th ≈ 1.96
    let q025 = result.quantile(0.025);
    let q975 = result.quantile(0.975);
    assert!((q025 - (-1.96)).abs() < 0.3);
    assert!((q975 - 1.96).abs() < 0.3);
}

// ===== Application =====

#[test]
fn test_binary_belief_converges() {
    let mut belief = BinaryBelief::uninformative("coin");
    // True probability is 0.7
    let mut rng = rng();
    for _ in 0..1000 {
        belief.observe(rng.gen::<f64>() < 0.7);
    }
    assert!((belief.expected_probability() - 0.7).abs() < 0.05);
}

#[test]
fn test_decision_with_clear_winner() {
    let mut d = Decision::new();
    d.add_option("safe", 5.0, 0.1);
    d.add_option("risky", 10.0, 5.0);
    let prob = d.prob_i_beats_j(0, 1);
    // Risky has higher mean but safe should win sometimes
    assert!(prob > 0.0 && prob < 1.0);
}
