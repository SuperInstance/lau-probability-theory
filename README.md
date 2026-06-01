# lau-probability-theory

A Rust library for **probability theory** — discrete and continuous distributions, limit theorems, Bayesian inference, hypothesis testing, MCMC sampling, copulas, and agent belief modeling.

## What This Does

This crate provides a comprehensive toolkit for probabilistic modeling and statistical inference:

- **Discrete distributions** — Bernoulli, Binomial, Poisson, Geometric, Uniform, Hypergeometric
- **Continuous distributions** — Normal, Exponential, Uniform, Gamma, Beta, Chi-squared, Student-t
- **Distribution traits** — unified `Distribution` trait with PDF/PMF, CDF, mean, variance, sampling
- **Limit theorems** — Law of Large Numbers (weak & strong), Central Limit Theorem verification
- **Bayesian inference** — conjugate priors (Beta-Binomial, Normal-Normal), posterior updates, credible intervals, Bayesian model comparison
- **Hypothesis testing** — Z-test, t-test, chi-squared test for proportions, with p-value computation
- **MCMC** — Metropolis-Hastings sampler with configurable proposal, burn-in, thinning, and diagnostics
- **Copulas** — Gaussian copula, Clayton copula, Independence copula, with CDF, sampling, and Kendall's tau
- **Agent beliefs** — `BinaryBelief` (Beta posterior), `ContinuousBelief` (Normal posterior), `Decision` with expected-utility maximization

## Key Idea

Probability is the mathematical language of uncertainty. This library treats distributions as first-class objects that implement a common `Distribution` trait — every distribution knows its density, cumulative function, moments, and can generate random samples.

The Bayesian modules implement the **conjugate prior** pattern: observing data updates a prior distribution into a posterior of the same family, in closed form. This enables real-time belief tracking without approximate inference.

The agent belief system layers on top: an agent holds a belief (probability distribution over an unknown quantity), updates it with observations, and makes decisions by maximizing expected utility under its posterior.

## Install

```toml
[dependencies]
lau-probability-theory = "0.1"
```

Requires **Rust 2021 edition**.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (vectors, matrices) |
| `serde` | Serialization of distributions and results |
| `rand` / `rand_distr` | Random sampling |

## Quick Start

### Working with distributions

```rust
use lau_probability_theory::continuous::Normal;
use lau_probability_theory::distributions::Distribution;

let normal = Normal::new(0.0, 1.0); // μ=0, σ=1
assert_eq!(normal.mean(), 0.0);
assert_eq!(normal.variance(), 1.0);

let pdf_at_zero = normal.pdf(0.0);   // ≈ 0.3989
let cdf_at_zero = normal.cdf(0.0);   // = 0.5

// Sample from the distribution
use rand::thread_rng;
let sample: f64 = normal.sample(&mut thread_rng());
```

### Bayesian inference

```rust
use lau_probability_theory::bayesian::BetaPosterior;

// Start with uniform prior: Beta(1, 1)
let prior = BetaPosterior::new(1.0, 1.0);

// Observe 8 successes, 2 failures
let posterior = prior.update(8, 2);

// Posterior is Beta(9, 3)
assert_eq!(posterior.alpha(), 9.0);
assert_eq!(posterior.beta(), 3.0);

// 95% credible interval
let (lower, upper) = posterior.credible_interval(0.95);
```

### Hypothesis testing

```rust
use lau_probability_theory::hypothesis_testing::{z_test, HypothesisTest};

// Test if sample mean ≠ 100 with known σ=15, n=50, x̄=105
let result = z_test(105.0, 100.0, 15.0, 50);
assert!(result.reject_at(0.05)); // Reject H₀ at α=0.05
let p_value = result.p_value();
```

### MCMC sampling

```rust
use lau_probability_theory::mcmc::MetropolisHastings;
use rand::SeedableRng;
use rand::rngs::StdRng;

let mut rng = StdRng::seed_from_u64(42);

// Sample from a bimodal target: log density of mixture
let log_target = |x: f64| {
    let a = -0.5 * (x - 2.0).powi(2);
    let b = -0.5 * (x + 2.0).powi(2);
    (a.exp() + b.exp()).ln()
};

let mh = MetropolisHastings::new(1.5, 5000, 3); // σ=1.5, burn-in=5000, thin=3
let result = mh.sample(log_target, 0.0, 10000, &mut rng);
println!("Mean: {:.3}", result.mean());
println!("Acceptance rate: {:.1}%", result.acceptance_rate * 100.0);
```

### Copulas

```rust
use lau_probability_theory::copulas::GaussianCopula;
use rand::thread_rng;

// Model dependence with ρ = 0.7
let copula = GaussianCopula::new(0.7);
let (u, v) = copula.sample(&mut thread_rng());

// Kendall's tau
let tau = copula.kendall_tau(); // = (2/π)arcsin(ρ)
```

### Agent beliefs & decisions

```rust
use lau_probability_theory::application::{BinaryBelief, Decision};

// Track belief about task success
let mut belief = BinaryBelief::uninformative("deploy_success");
belief.observe(true);  // success
belief.observe(true);
belief.observe(false); // failure

println!("P(success) = {:.2}", belief.expected_probability());

// Choose between options with uncertain utilities
let mut decision = Decision::new();
decision.add_option("FastDeploy", 8.0, 3.0);
decision.add_option("SafeDeploy", 6.0, 1.0);

let best = decision.best_option();
let p_fast_beats_safe = decision.prob_i_beats_j(0, 1);
```

## API Reference

### `distributions` — Core Traits

| Trait | Methods |
|-------|---------|
| `Distribution` | `pdf(&self, x)`, `cdf(&self, x)`, `mean()`, `variance()`, `sample(&self, rng)` |

### `discrete` — Discrete Distributions

| Distribution | Parameters | Support |
|-------------|-----------|---------|
| `Bernoulli` | `p` | {0, 1} |
| `Binomial` | `n, p` | {0, ..., n} |
| `Poisson` | `λ` | {0, 1, 2, ...} |
| `Geometric` | `p` | {1, 2, 3, ...} |
| `DiscreteUniform` | `a, b` | {a, ..., b} |
| `Hypergeometric` | `N, K, n` | {max(0, n+K-N), ..., min(K, n)} |

### `continuous` — Continuous Distributions

| Distribution | Parameters | PDF | CDF |
|-------------|-----------|-----|-----|
| `Normal` | `μ, σ` | Gaussian bell curve | Error function |
| `Exponential` | `λ` | `λe^{-λx}` | `1 - e^{-λx}` |
| `Uniform` | `a, b` | `1/(b-a)` | Step |
| `Gamma` | `α, β` | `x^{α-1}e^{-x/β}` | Regularized incomplete gamma |
| `Beta` | `α, β` | Power function on [0,1] | Regularized incomplete beta |
| `ChiSquared` | `k` | Gamma(k/2, 2) | — |
| `StudentT` | `ν` | Heavy-tailed bell | — |

Plus: `standard_normal_cdf`, `standard_normal_inv` (inverse CDF / quantile).

### `bayesian` — Bayesian Inference

| Item | Description |
|------|-------------|
| `BetaPosterior` | Beta-Binomial conjugate model with `update(successes, failures)`, `mean()`, `credible_interval()` |
| `normal_normal_update` | Normal-Normal conjugate update: returns posterior `(μ', σ')` |
| `normal_credible_interval` | Credible interval for Normal posterior |

### `hypothesis_testing` — Statistical Tests

| Test | Function | Description |
|------|----------|-------------|
| Z-test | `z_test(x̄, μ₀, σ, n)` | Known variance, large sample |
| t-test | `t_test(x̄, μ₀, s, n)` | Unknown variance, small sample |
| Chi-squared | `chi_squared_test(observed, expected)` | Goodness of fit |

All return a `HypothesisTestResult` with `p_value()`, `reject_at(α)`, and `statistic()`.

### `limit_theorems` — LLN & CLT

| Function | Description |
|----------|-------------|
| `law_of_large_numbers` | Simulate sample means converging to `E[X]` |
| `central_limit_theorem` | Show standardized means → N(0,1) |

### `mcmc` — Markov Chain Monte Carlo

| Item | Description |
|------|-------------|
| `MetropolisHastings` | Configurable `proposal_sigma`, `burn_in`, `thin` |
| `MCMCResult` | `samples`, `acceptance_rate`, `total_steps`, `mean()`, `variance()`, `quantile()` |

### `copulas` — Dependence Modeling

| Copula | Parameters | Methods |
|--------|-----------|---------|
| `GaussianCopula` | `ρ ∈ [-1, 1]` | `cdf(u, v)`, `sample(rng)`, `kendall_tau()` |
| `ClaytonCopula` | `θ ≥ -1` | Same interface |
| `IndependenceCopula` | — | `C(u,v) = uv` |

### `application` — Agent Beliefs

| Item | Description |
|------|-------------|
| `BinaryBelief` | Beta-posterior tracker for binary outcomes: `observe(bool)`, `expected_probability()`, `uncertainty()`, `credible_interval_95()` |
| `ContinuousBelief` | Normal-posterior tracker for continuous quantities |
| `Decision` | Expected-utility maximization over uncertain options, `prob_i_beats_j()` for pairwise comparisons |

## How It Works

### Conjugate priors

The Beta-Binomial model: if the prior is `Beta(α, β)` and we observe `s` successes and `f` failures, the posterior is `Beta(α + s, β + f)`. No integration required — just addition.

The Normal-Normal model: prior `N(μ₀, σ₀²)`, likelihood `N(x | μ, σ²)`, posterior `N(μ', σ'²)` where:

```
1/σ'² = 1/σ₀² + n/σ²
μ' = σ'² · (μ₀/σ₀² + nx̄/σ²)
```

### Metropolis-Hastings

At each step:
1. Propose `x' = x + ε`, where `ε ~ N(0, σ²)`
2. Compute acceptance ratio `α = p(x') / p(x)` (in log space)
3. Accept if `α ≥ 1` or with probability `α` otherwise

Burn-in discards early samples before the chain reaches stationarity. Thinning reduces autocorrelation by keeping every k-th sample.

### Copulas

A copula `C: [0,1]² → [0,1]` decouples marginal distributions from the dependence structure via Sklar's theorem: any joint distribution `F(x,y)` can be written as `C(F₁(x), F₂(y))`.

The Gaussian copula with correlation `ρ` captures linear dependence. The Clayton copula with parameter `θ` captures asymmetric lower-tail dependence. Kendall's tau provides a rank-based dependence measure: for Gaussian, `τ = (2/π)arcsin(ρ)`.

## The Math

**Central Limit Theorem**: if `X₁, ..., Xₙ` are i.i.d. with mean `μ` and variance `σ²`, then:

```
√n · (X̄ - μ) / σ → N(0, 1)  as n → ∞
```

**Bayes' theorem**: the posterior is proportional to the likelihood times the prior:

```
P(θ | data) ∝ P(data | θ) · P(θ)
```

For conjugate families, the normalization constant is known analytically.

**Metropolis-Hastings** targets a distribution `π(x)` by constructing a Markov chain whose stationary distribution is `π`. The acceptance probability ensures **detailed balance**: `π(x)P(x→x') = π(x')P(x'→x)`.

**Copulas and Sklar's theorem**: for any continuous joint CDF `F` with marginals `F₁, F₂`, there exists a unique copula `C` such that `F(x,y) = C(F₁(x), F₂(y))`.

## License

MIT
