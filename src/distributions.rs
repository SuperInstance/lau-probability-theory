use serde::{Deserialize, Serialize};

/// Core trait for all probability distributions.
pub trait Distribution: Send + Sync {
    /// Probability density (continuous) or probability mass (discrete) at x.
    fn pdf(&self, x: f64) -> f64;
    /// Cumulative distribution function.
    fn cdf(&self, x: f64) -> f64;
    /// Mean of the distribution.
    fn mean(&self) -> f64;
    /// Variance of the distribution.
    fn variance(&self) -> f64;
    /// Sample a random value from this distribution.
    fn sample(&self, rng: &mut impl rand::Rng) -> f64;
    /// Moment generating function evaluated at t.
    fn mgf(&self, t: f64) -> f64;
}

/// Summary statistics for a sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleStats {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return SampleStats { n: 0, mean: 0.0, variance: 0.0, std_dev: 0.0, min: 0.0, max: 0.0 };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        SampleStats {
            n,
            mean,
            variance,
            std_dev: variance.sqrt(),
            min,
            max,
        }
    }
}
