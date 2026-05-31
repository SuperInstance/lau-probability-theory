# lau-probability-theory

Probability theory library for Rust — distributions, limit theorems, Bayesian inference, hypothesis testing, and MCMC.

## Features

### Discrete Distributions
- **Bernoulli**, **Binomial**, **Poisson**, **Geometric**, **Hypergeometric**
- Full PMF, CDF, mean, variance, sampling, and moment generating functions

### Continuous Distributions
- **Uniform**, **Normal**, **Exponential**, **Gamma**, **Beta**
- Full PDF, CDF, mean, variance, sampling, and MGFs

### Limit Theorems
- **Central Limit Theorem** verification via sampling experiments
- **Law of Large Numbers** verification
- Kolmogorov-Smirnov statistic computation

### Copulas
- **Gaussian** copula with configurable correlation
- **Clayton** copula for lower-tail dependence
- **Independence** copula
- Kendall's tau for each copula family

### Bayesian Inference
- Conjugate priors: **Normal-Normal**, **Beta-Binomial**, **Gamma-Poisson**
- Posterior updates and credible intervals
- Sequential Bayesian updating

### Hypothesis Testing
- **z-test** (known variance)
- **t-test** (unknown variance)
- **Chi-squared** goodness-of-fit test

### Markov Chain Monte Carlo
- **Metropolis-Hastings** sampler with configurable proposal, burn-in, and thinning
- Posterior summary statistics and quantiles

### Application: Agent Uncertainty
- `BinaryBelief` — probabilistic modeling of binary outcomes with Beta priors
- `ContinuousBelief` — Normal priors for continuous quantities
- `Decision` — compare options under uncertainty with probabilistic dominance

## Usage

```rust
use lau_probability_theory::discrete::{Bernoulli, Binomial};
use lau_probability_theory::continuous::Normal;
use lau_probability_theory::distributions::Distribution;
use lau_probability_theory::bayesian::{beta_binomial_update, BetaPosterior};

// Create and use distributions
let binom = Binomial::new(10, 0.5);
println!("Mean: {}", binom.mean());

// Bayesian updating
let prior = BetaPosterior::new(1.0, 1.0); // Uniform prior
let posterior = prior.update(7, 3);       // 7 successes, 3 failures
println!("Posterior mean: {}", posterior.mean());
```

## Dependencies
- `serde` — serialization
- `nalgebra` — linear algebra
- `rand` — random sampling

## License
MIT
