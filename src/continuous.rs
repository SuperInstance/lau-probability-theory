use crate::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Continuous Uniform distribution on [a, b].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uniform {
    pub a: f64,
    pub b: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Self {
        assert!(a < b, "a must be less than b");
        Uniform { a, b }
    }
}

impl Distribution for Uniform {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b { 0.0 } else { 1.0 / (self.b - self.a) }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.a { 0.0 } else if x >= self.b { 1.0 } else { (x - self.a) / (self.b - self.a) }
    }
    fn mean(&self) -> f64 { (self.a + self.b) / 2.0 }
    fn variance(&self) -> f64 { (self.b - self.a).powi(2) / 12.0 }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        self.a + rng.gen::<f64>() * (self.b - self.a)
    }
    fn mgf(&self, t: f64) -> f64 {
        if t.abs() < 1e-10 { 1.0 } else {
            (t * self.b).exp() - (t * self.a).exp() / (t * (self.b - self.a))
        }
    }
}

/// Normal (Gaussian) distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Normal { mu, sigma }
    }

    pub fn standard() -> Self { Normal { mu: 0.0, sigma: 1.0 } }
}

impl Distribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * std::f64::consts::PI).sqrt())
    }
    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mu) / (self.sigma * std::f64::consts::SQRT_2)))
    }
    fn mean(&self) -> f64 { self.mu }
    fn variance(&self) -> f64 { self.sigma * self.sigma }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        // Box-Muller
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        self.mu + self.sigma * z0
    }
    fn mgf(&self, t: f64) -> f64 {
        (self.mu * t + 0.5 * self.sigma * self.sigma * t * t).exp()
    }
}

/// Error function approximation (Abramowitz and Stegun).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Exponential distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive");
        Exponential { lambda }
    }
}

impl Distribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.lambda * (-self.lambda * x).exp() }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { 1.0 - (-self.lambda * x).exp() }
    }
    fn mean(&self) -> f64 { 1.0 / self.lambda }
    fn variance(&self) -> f64 { 1.0 / (self.lambda * self.lambda) }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        -rng.gen::<f64>().ln() / self.lambda
    }
    fn mgf(&self, t: f64) -> f64 {
        assert!(t < self.lambda, "t must be less than lambda");
        self.lambda / (self.lambda - t)
    }
}

/// Gamma distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gamma {
    pub shape: f64,
    pub rate: f64,
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Self {
        assert!(shape > 0.0, "shape must be positive");
        assert!(rate > 0.0, "rate must be positive");
        Gamma { shape, rate }
    }
}

impl Distribution for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let k = self.shape;
        let theta = 1.0 / self.rate;
        (x.powf(k - 1.0) * (-x / theta).exp()) / (theta.powf(k) * gamma_fn(k))
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        // Regularized lower incomplete gamma function
        regularized_gamma_p(self.shape, self.rate * x)
    }
    fn mean(&self) -> f64 { self.shape / self.rate }
    fn variance(&self) -> f64 { self.shape / (self.rate * self.rate) }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        // Marsaglia and Tsang's method for shape >= 1
        if self.shape >= 1.0 {
            let d = self.shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let mut x: f64;
                let mut v: f64;
                loop {
                    x = Normal::standard().sample(rng);
                    v = 1.0 + c * x;
                    if v > 0.0 { break; }
                }
                v = v * v * v;
                let u: f64 = rng.gen();
                if u < 1.0 - 0.0331 * x * x * x * x { return d * v / self.rate; }
                if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) { return d * v / self.rate; }
            }
        } else {
            // For shape < 1, use the relation: Gamma(k) = Gamma(k+1) * U^(1/k)
            let g = Gamma::new(self.shape + 1.0, self.rate);
            let val = g.sample(rng);
            val * rng.gen::<f64>().powf(1.0 / self.shape)
        }
    }
    fn mgf(&self, t: f64) -> f64 {
        assert!(t < self.rate, "t must be less than rate");
        (self.rate / (self.rate - t)).powf(self.shape)
    }
}

/// Beta distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0, "alpha must be positive");
        assert!(beta > 0.0, "beta must be positive");
        Beta { alpha, beta }
    }
}

impl Distribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 { return 0.0; }
        let num = x.powf(self.alpha - 1.0) * (1.0 - x).powf(self.beta - 1.0);
        let den = beta_fn(self.alpha, self.beta);
        num / den
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        regularized_beta(x, self.alpha, self.beta)
    }
    fn mean(&self) -> f64 { self.alpha / (self.alpha + self.beta) }
    fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let x = Gamma::new(self.alpha, 1.0).sample(rng);
        let y = Gamma::new(self.beta, 1.0).sample(rng);
        x / (x + y)
    }
    fn mgf(&self, _t: f64) -> f64 {
        // No simple closed form
        f64::NAN
    }
}

/// Log-gamma function (Stirling-based for large values, Lanczos for small).
pub fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 { return f64::NAN; }
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - log_gamma(1.0 - x);
    }
    let z = x - 1.0;
    let g = 7.0;
    let coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let mut s = coef[0];
    for i in 1..coef.len() {
        s += coef[i] / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
}

/// Gamma function.
pub fn gamma_fn(x: f64) -> f64 {
    log_gamma(x).exp()
}

/// Beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b).
pub fn beta_fn(a: f64, b: f64) -> f64 {
    (log_gamma(a) + log_gamma(b) - log_gamma(a + b)).exp()
}

/// Regularized incomplete gamma function P(a, x) via series expansion.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-12 { break; }
    }
    sum * (-x).exp() * x.powf(a) / gamma_fn(a)
}

/// Regularized incomplete beta function via series.
pub fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    // Use symmetry if x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_beta(1.0 - x, b, a);
    }
    // Power series: I_x(a,b) = x^a * (1-x)^b / (a * B(a,b)) * sum_{k=0}^inf prod
    let mut result = 0.0;
    let mut term = 1.0;
    for k in 0..200 {
        if k > 0 {
            term *= (a + b + k as f64 - 1.0) * x / ((a + k as f64) * k as f64);
        }
        result += term;
        if k > 0 && term.abs() < result.abs() * 1e-12 { break; }
    }
    x.powf(a) * (1.0 - x).powf(b) * result / (a * beta_fn(a, b))
}

/// Standard normal CDF.
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal (quantile) via rational approximation.
pub fn standard_normal_inv(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if p == 0.5 { return 0.0; }

    let (p, flip) = if p < 0.5 { (p, true) } else { (1.0 - p, false) };

    let t = (-2.0 * p.ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let mut x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if flip { x = -x; }
    x
}
