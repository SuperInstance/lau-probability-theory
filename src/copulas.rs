use serde::{Deserialize, Serialize};
use rand::Rng;
use crate::distributions::Distribution;
use crate::continuous::{Normal, standard_normal_cdf, standard_normal_inv};

/// Gaussian (normal) copula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianCopula {
    pub rho: f64,
}

impl GaussianCopula {
    pub fn new(rho: f64) -> Self {
        assert!((-1.0..=1.0).contains(&rho), "rho must be in [-1, 1]");
        GaussianCopula { rho }
    }

    /// CDF of the Gaussian copula at (u, v).
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 { return 0.0; }
        if u >= 1.0 && v >= 1.0 { return 1.0; }
        let x = standard_normal_inv(u);
        let y = standard_normal_inv(v);
        // Use numerical integration for bivariate normal CDF
        bivariate_normal_cdf(x, y, self.rho)
    }

    /// Sample from the Gaussian copula.
    pub fn sample(&self, rng: &mut impl Rng) -> (f64, f64) {
        let z1 = Normal::standard().sample(rng);
        let z2 = self.rho * z1 + (1.0 - self.rho * self.rho).sqrt() * Normal::standard().sample(rng);
        (standard_normal_cdf(z1), standard_normal_cdf(z2))
    }

    /// Kendall's tau for Gaussian copula.
    pub fn kendall_tau(&self) -> f64 {
        (2.0 / std::f64::consts::PI) * self.rho.asin()
    }
}

/// Clayton copula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaytonCopula {
    pub theta: f64,
}

impl ClaytonCopula {
    pub fn new(theta: f64) -> Self {
        assert!(theta >= -1.0, "theta must be >= -1");
        ClaytonCopula { theta }
    }

    /// CDF of the Clayton copula.
    pub fn cdf(&self, u: f64, v: f64) -> f64 {
        if u <= 0.0 || v <= 0.0 { return 0.0; }
        if u >= 1.0 && v >= 1.0 { return 1.0; }
        if self.theta == 0.0 { return u * v } // Independence
        let t = self.theta;
        (u.powf(-t) + v.powf(-t) - 1.0).powf(-1.0 / t).max(0.0)
    }

    /// Sample from the Clayton copula using conditional method.
    pub fn sample(&self, rng: &mut impl Rng) -> (f64, f64) {
        if self.theta == 0.0 {
            return (rng.gen::<f64>(), rng.gen::<f64>());
        }
        let u: f64 = rng.gen();
        let w: f64 = rng.gen();
        let t = self.theta;
        let v = (1.0 + u.powf(-t) * (w.powf(-t / (1.0 + t)) - 1.0)).powf(-1.0 / t);
        (u, v.max(0.0).min(1.0))
    }

    /// Kendall's tau for Clayton copula.
    pub fn kendall_tau(&self) -> f64 {
        self.theta / (self.theta + 2.0)
    }
}

/// Independence copula (product copula).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependenceCopula;

impl IndependenceCopula {
    pub fn new() -> Self { IndependenceCopula }
    pub fn cdf(&self, u: f64, v: f64) -> f64 { u * v }
    pub fn sample(&self, rng: &mut impl Rng) -> (f64, f64) {
        (rng.gen::<f64>(), rng.gen::<f64>())
    }
    pub fn kendall_tau(&self) -> f64 { 0.0 }
}

/// Approximate bivariate normal CDF.
fn bivariate_normal_cdf(x: f64, y: f64, rho: f64) -> f64 {
    if rho.abs() < 1e-10 {
        return standard_normal_cdf(x) * standard_normal_cdf(y);
    }
    if (rho - 1.0).abs() < 1e-10 {
        return standard_normal_cdf(x.min(y));
    }
    if (rho + 1.0).abs() < 1e-10 {
        if x + y <= 0.0 {
            return standard_normal_cdf(x) - standard_normal_cdf(-y);
        }
        return standard_normal_cdf(y) - standard_normal_cdf(-x);
    }

    // Numerical integration
    let n = 100;
    let dx = x / n as f64;
    let mut sum = 0.0;
    for i in 0..n {
        let xi = (i as f64 + 0.5) * dx;
        let phi_xi = (-0.5 * xi * xi).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let z = (y - rho * xi) / (1.0 - rho * rho).sqrt();
        sum += phi_xi * standard_normal_cdf(z) * dx;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use approx::assert_relative_eq;

    #[test]
    fn test_independence_copula() {
        let cop = IndependenceCopula::new();
        assert_relative_eq!(cop.cdf(0.5, 0.5), 0.25);
        assert_relative_eq!(cop.cdf(0.0, 0.5), 0.0);
        assert_relative_eq!(cop.cdf(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_clayton_copula_bounds() {
        let cop = ClaytonCopula::new(2.0);
        assert_relative_eq!(cop.cdf(0.0, 0.5), 0.0);
        assert_relative_eq!(cop.cdf(1.0, 1.0), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_clayton_kendall_tau() {
        let cop = ClaytonCopula::new(2.0);
        assert_relative_eq!(cop.kendall_tau(), 0.5);
    }

    #[test]
    fn test_gaussian_kendall_tau() {
        let cop = GaussianCopula::new(0.0);
        assert_relative_eq!(cop.kendall_tau(), 0.0);
    }

    #[test]
    fn test_gaussian_copula_independence() {
        let cop = GaussianCopula::new(0.0);
        // When rho=0, C(u,v) ≈ u*v
        let val = cop.cdf(0.5, 0.5);
        assert!((val - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_copula_samples_in_unit_square() {
        let mut rng = StdRng::seed_from_u64(42);
        let cop = GaussianCopula::new(0.5);
        for _ in 0..100 {
            let (u, v) = cop.sample(&mut rng);
            assert!(u >= 0.0 && u <= 1.0);
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_clayton_copula_samples_in_unit_square() {
        let mut rng = StdRng::seed_from_u64(42);
        let cop = ClaytonCopula::new(1.5);
        for _ in 0..100 {
            let (u, v) = cop.sample(&mut rng);
            assert!(u >= 0.0 && u <= 1.0);
            assert!(v >= 0.0 && v <= 1.0);
        }
    }
}
