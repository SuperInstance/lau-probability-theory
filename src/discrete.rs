use crate::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Bernoulli distribution: P(X=1) = p, P(X=0) = 1-p.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bernoulli {
    pub p: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
        Bernoulli { p }
    }
}

impl Distribution for Bernoulli {
    fn pdf(&self, x: f64) -> f64 {
        if x == 1.0 { self.p } else if x == 0.0 { 1.0 - self.p } else { 0.0 }
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else if x < 1.0 { 1.0 - self.p } else { 1.0 }
    }
    fn mean(&self) -> f64 { self.p }
    fn variance(&self) -> f64 { self.p * (1.0 - self.p) }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        if rng.gen::<f64>() < self.p { 1.0 } else { 0.0 }
    }
    fn mgf(&self, t: f64) -> f64 {
        1.0 - self.p + self.p * t.exp()
    }
}

/// Binomial distribution: number of successes in n trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binomial {
    pub n: u64,
    pub p: f64,
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
        Binomial { n, p }
    }

    fn log_factorial(k: u64) -> f64 {
        if k == 0 { 0.0 } else {
            let mut s = 0.0;
            for i in 1..=k { s += (i as f64).ln(); }
            s
        }
    }

    pub fn pmf(&self, k: u64) -> f64 {
        if k > self.n { return 0.0; }
        let log_comb = Self::log_factorial(self.n) - Self::log_factorial(k) - Self::log_factorial(self.n - k);
        (log_comb + (k as f64) * self.p.ln() + ((self.n - k) as f64) * (1.0 - self.p).ln()).exp()
    }
}

impl Distribution for Binomial {
    fn pdf(&self, x: f64) -> f64 { self.pmf(x as u64) }
    fn cdf(&self, x: f64) -> f64 {
        let k = x.floor() as u64;
        (0..=k.min(self.n)).map(|i| self.pmf(i)).sum()
    }
    fn mean(&self) -> f64 { self.n as f64 * self.p }
    fn variance(&self) -> f64 { self.n as f64 * self.p * (1.0 - self.p) }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mut successes = 0u64;
        for _ in 0..self.n {
            if rng.gen::<f64>() < self.p { successes += 1; }
        }
        successes as f64
    }
    fn mgf(&self, t: f64) -> f64 {
        (1.0 - self.p + self.p * t.exp()).powi(self.n as i32)
    }
}

/// Poisson distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive");
        Poisson { lambda }
    }

    pub fn pmf(&self, k: u64) -> f64 {
        let log_pmf = k as f64 * self.lambda.ln() - self.lambda - Self::log_factorial(k);
        log_pmf.exp()
    }

    fn log_factorial(k: u64) -> f64 {
        (1..=k).map(|i| (i as f64).ln()).sum::<f64>()
    }
}

impl Distribution for Poisson {
    fn pdf(&self, x: f64) -> f64 { self.pmf(x as u64) }
    fn cdf(&self, x: f64) -> f64 {
        let k = x.floor() as u64;
        (0..=k).map(|i| self.pmf(i)).sum()
    }
    fn mean(&self) -> f64 { self.lambda }
    fn variance(&self) -> f64 { self.lambda }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        // Knuth's algorithm
        let l = (-self.lambda).exp();
        let mut k = 0u64;
        let mut p = 1.0;
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l { break; }
        }
        (k - 1) as f64
    }
    fn mgf(&self, t: f64) -> f64 {
        (self.lambda * (t.exp() - 1.0)).exp()
    }
}

/// Geometric distribution: number of trials until first success.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geometric {
    pub p: f64,
}

impl Geometric {
    pub fn new(p: f64) -> Self {
        assert!(p > 0.0 && p <= 1.0, "p must be in (0, 1]");
        Geometric { p }
    }

    pub fn pmf(&self, k: u64) -> f64 {
        if k == 0 { return 0.0; }
        self.p * (1.0 - self.p).powi(k as i32 - 1)
    }
}

impl Distribution for Geometric {
    fn pdf(&self, x: f64) -> f64 { self.pmf(x as u64) }
    fn cdf(&self, x: f64) -> f64 {
        if x < 1.0 { return 0.0; }
        1.0 - (1.0 - self.p).powi(x.floor() as i32)
    }
    fn mean(&self) -> f64 { 1.0 / self.p }
    fn variance(&self) -> f64 { (1.0 - self.p) / (self.p * self.p) }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mut trials = 1u64;
        while rng.gen::<f64>() >= self.p { trials += 1; }
        trials as f64
    }
    fn mgf(&self, t: f64) -> f64 {
        let et = t.exp();
        self.p * et / (1.0 - (1.0 - self.p) * et)
    }
}

/// Hypergeometric distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypergeometric {
    pub population: u64,
    pub successes: u64,
    pub draws: u64,
}

impl Hypergeometric {
    pub fn new(population: u64, successes: u64, draws: u64) -> Self {
        assert!(successes <= population, "successes must be <= population");
        assert!(draws <= population, "draws must be <= population");
        Hypergeometric { population, successes, draws }
    }

    fn log_comb(n: u64, k: u64) -> f64 {
        if k > n { return f64::NEG_INFINITY; }
        let mut s = 0.0;
        for i in 0..k.min(n - k) {
            s += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
        }
        s
    }

    pub fn pmf(&self, k: u64) -> f64 {
        if k > self.draws || k > self.successes { return 0.0; }
        let num = Self::log_comb(self.successes, k) + Self::log_comb(self.population - self.successes, self.draws - k);
        let den = Self::log_comb(self.population, self.draws);
        (num - den).exp()
    }
}

impl Distribution for Hypergeometric {
    fn pdf(&self, x: f64) -> f64 { self.pmf(x as u64) }
    fn cdf(&self, x: f64) -> f64 {
        let k = x.floor() as u64;
        (0..=k).map(|i| self.pmf(i)).sum()
    }
    fn mean(&self) -> f64 {
        self.draws as f64 * self.successes as f64 / self.population as f64
    }
    fn variance(&self) -> f64 {
        let n = self.draws as f64;
        let k = self.successes as f64;
        let m = self.population as f64;
        n * (k / m) * ((m - k) / m) * ((m - n) / (m - 1.0))
    }
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let mut successes_drawn = 0u64;
        let mut remaining_successes = self.successes;
        let mut remaining_population = self.population;
        for _ in 0..self.draws {
            if rng.gen::<f64>() < remaining_successes as f64 / remaining_population as f64 {
                successes_drawn += 1;
                remaining_successes -= 1;
            }
            remaining_population -= 1;
        }
        successes_drawn as f64
    }
    fn mgf(&self, _t: f64) -> f64 {
        // No simple closed form for hypergeometric MGF
        f64::NAN
    }
}
