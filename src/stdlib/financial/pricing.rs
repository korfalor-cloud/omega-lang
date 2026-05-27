/// Financial mathematics: options pricing, portfolio metrics, risk measures.

use std::f64::consts::{PI, E};

/// Normal distribution CDF (Abramowitz & Stegun approximation).
pub fn norm_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

    0.5 * (1.0 + sign * y)
}

/// Normal distribution PDF.
pub fn norm_pdf(x: f64) -> f64 {
    (-x * x / 2.0).exp() / (2.0 * PI).sqrt()
}

/// Black-Scholes European call option price.
pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (s - k).max(0.0);
    }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

/// Black-Scholes European put option price.
pub fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return (k - s).max(0.0);
    }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1)
}

/// Option Greeks.
#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Compute Greeks for a European call option.
pub fn call_greeks(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> Greeks {
    if t <= 0.0 || sigma <= 0.0 {
        return Greeks { delta: if s > k { 1.0 } else { 0.0 }, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 };
    }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let delta = norm_cdf(d1);
    let gamma = norm_pdf(d1) / (s * sigma * t.sqrt());
    let theta = -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt())
        - r * k * (-r * t).exp() * norm_cdf(d2);
    let vega = s * norm_pdf(d1) * t.sqrt() / 100.0; // per 1% move
    let rho = k * t * (-r * t).exp() * norm_cdf(d2) / 100.0;

    Greeks { delta, gamma, theta, vega, rho }
}

/// Compute Greeks for a European put option.
pub fn put_greeks(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> Greeks {
    if t <= 0.0 || sigma <= 0.0 {
        return Greeks { delta: if s < k { -1.0 } else { 0.0 }, gamma: 0.0, theta: 0.0, vega: 0.0, rho: 0.0 };
    }
    let d1 = ((s / k).ln() + (r + sigma * sigma / 2.0) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();

    let delta = norm_cdf(d1) - 1.0;
    let gamma = norm_pdf(d1) / (s * sigma * t.sqrt());
    let theta = -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt())
        + r * k * (-r * t).exp() * norm_cdf(-d2);
    let vega = s * norm_pdf(d1) * t.sqrt() / 100.0;
    let rho = -k * t * (-r * t).exp() * norm_cdf(-d2) / 100.0;

    Greeks { delta, gamma, theta, vega, rho }
}

/// Implied volatility using Newton-Raphson method.
pub fn implied_volatility(market_price: f64, s: f64, k: f64, t: f64, r: f64, is_call: bool) -> Option<f64> {
    let mut sigma = 0.3; // Initial guess
    for _ in 0..100 {
        let price = if is_call {
            black_scholes_call(s, k, t, r, sigma)
        } else {
            black_scholes_put(s, k, t, r, sigma)
        };
        let vega = if is_call {
            call_greeks(s, k, t, r, sigma).vega * 100.0
        } else {
            put_greeks(s, k, t, r, sigma).vega * 100.0
        };

        if vega.abs() < 1e-10 {
            return None;
        }

        let diff = price - market_price;
        sigma -= diff / vega;

        if diff.abs() < 1e-8 {
            return Some(sigma);
        }
        if sigma <= 0.0 {
            sigma = 0.001;
        }
    }
    Some(sigma)
}

/// Binomial option pricing model.
pub fn binomial_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, steps: usize, is_call: bool) -> f64 {
    let dt = t / steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((r * dt).exp() - d) / (u - d);
    let discount = (-r * dt).exp();

    let mut prices = vec![0.0f64; steps + 1];
    for i in 0..=steps {
        let spot = s * u.powi(i as i32) * d.powi((steps - i) as i32);
        prices[i] = if is_call { (spot - k).max(0.0) } else { (k - spot).max(0.0) };
    }

    for step in (0..steps).rev() {
        for i in 0..=step {
            prices[i] = discount * (p * prices[i + 1] + (1.0 - p) * prices[i]);
        }
    }

    prices[0]
}

/// Monte Carlo option pricing.
pub fn monte_carlo_call(s: f64, k: f64, t: f64, r: f64, sigma: f64, simulations: usize) -> f64 {
    let mut seed: u64 = 42;
    let mut total = 0.0;
    let dt = t;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    for _ in 0..simulations {
        let z = box_muller(&mut seed);
        let st = s * (drift + vol * z).exp();
        total += (st - k).max(0.0);
    }

    (-r * t).exp() * total / simulations as f64
}

fn box_muller(seed: &mut u64) -> f64 {
    let u1 = pseudo_random(seed);
    let u2 = pseudo_random(seed);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn pseudo_random(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (1u64 << 31) as f64
}

/// Portfolio return.
pub fn portfolio_return(returns: &[f64], weights: &[f64]) -> f64 {
    returns.iter().zip(weights.iter()).map(|(r, w)| r * w).sum()
}

/// Portfolio volatility.
pub fn portfolio_volatility(weights: &[f64], covariance: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * covariance[i][j];
        }
    }
    variance.max(0.0).sqrt()
}

/// Sharpe ratio.
pub fn sharpe_ratio(portfolio_return: f64, risk_free_rate: f64, volatility: f64) -> f64 {
    if volatility < 1e-10 {
        return 0.0;
    }
    (portfolio_return - risk_free_rate) / volatility
}

/// Value at Risk (parametric).
pub fn value_at_risk(portfolio_value: f64, mean_return: f64, volatility: f64, confidence: f64) -> f64 {
    let z = inverse_norm_cdf(1.0 - confidence);
    portfolio_value * (mean_return + z * volatility)
}

/// Conditional Value at Risk (Expected Shortfall).
pub fn conditional_var(portfolio_value: f64, mean_return: f64, volatility: f64, confidence: f64) -> f64 {
    let z = inverse_norm_cdf(1.0 - confidence);
    let phi_z = norm_pdf(z);
    portfolio_value * (mean_return - volatility * phi_z / (1.0 - confidence))
}

/// Inverse normal CDF (rational approximation).
fn inverse_norm_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5])
            / ((((b[0]*q+b[1])*q+b[2])*q+b[3])*q+b[4]+1.0);
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q
            / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5])
            / ((((b[0]*q+b[1])*q+b[2])*q+b[3])*q+b[4]+1.0);
    }
}

/// Compound annual growth rate.
pub fn cagr(begin_value: f64, end_value: f64, years: f64) -> f64 {
    if begin_value <= 0.0 || years <= 0.0 {
        return 0.0;
    }
    (end_value / begin_value).powf(1.0 / years) - 1.0
}

/// Maximum drawdown from a price series.
pub fn max_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    let mut peak = prices[0];
    let mut max_dd = 0.0;
    for &price in prices {
        peak = peak.max(price);
        let dd = (peak - price) / peak;
        max_dd = max_dd.max(dd);
    }
    max_dd
}

/// Sortino ratio.
pub fn sortino_ratio(portfolio_return: f64, risk_free_rate: f64, downside_deviation: f64) -> f64 {
    if downside_deviation < 1e-10 {
        return 0.0;
    }
    (portfolio_return - risk_free_rate) / downside_deviation
}

/// Downside deviation of returns.
pub fn downside_deviation(returns: &[f64], threshold: f64) -> f64 {
    let n = returns.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let sum: f64 = returns.iter()
        .map(|r| (r - threshold).min(0.0).powi(2))
        .sum();
    (sum / n).sqrt()
}

/// Information ratio.
pub fn information_ratio(portfolio_return: f64, benchmark_return: f64, tracking_error: f64) -> f64 {
    if tracking_error < 1e-10 {
        return 0.0;
    }
    (portfolio_return - benchmark_return) / tracking_error
}

/// Bond price from yield.
pub fn bond_price(face_value: f64, coupon_rate: f64, yield_rate: f64, periods: usize) -> f64 {
    let coupon = face_value * coupon_rate;
    let mut price = 0.0;
    for t in 1..=periods {
        price += coupon / (1.0 + yield_rate).powi(t as i32);
    }
    price += face_value / (1.0 + yield_rate).powi(periods as i32);
    price
}

/// Bond duration (Macaulay).
pub fn bond_duration(face_value: f64, coupon_rate: f64, yield_rate: f64, periods: usize) -> f64 {
    let coupon = face_value * coupon_rate;
    let price = bond_price(face_value, coupon_rate, yield_rate, periods);
    if price < 1e-10 {
        return 0.0;
    }

    let mut weighted_sum = 0.0;
    for t in 1..=periods {
        let cf = if t == periods { coupon + face_value } else { coupon };
        weighted_sum += t as f64 * cf / (1.0 + yield_rate).powi(t as i32);
    }
    weighted_sum / price
}

/// Bond convexity.
pub fn bond_convexity(face_value: f64, coupon_rate: f64, yield_rate: f64, periods: usize) -> f64 {
    let coupon = face_value * coupon_rate;
    let price = bond_price(face_value, coupon_rate, yield_rate, periods);
    if price < 1e-10 {
        return 0.0;
    }

    let mut convexity_sum = 0.0;
    for t in 1..=periods {
        let cf = if t == periods { coupon + face_value } else { coupon };
        let pv = cf / (1.0 + yield_rate).powi(t as i32);
        convexity_sum += t as f64 * (t as f64 + 1.0) * pv;
    }
    convexity_sum / (price * (1.0 + yield_rate).powi(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_norm_cdf() {
        assert!(approx_eq(norm_cdf(0.0), 0.5, 1e-6));
        assert!(approx_eq(norm_cdf(1.96), 0.975, 1e-3));
        assert!(norm_cdf(-10.0) < 0.001);
    }

    #[test]
    fn test_black_scholes() {
        let call = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        let put = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(call > 0.0);
        assert!(put > 0.0);
        // Put-call parity: C - P = S - K*e^(-rT)
        let parity = call - put;
        let expected = 100.0 - 100.0 * (-0.05_f64).exp();
        assert!(approx_eq(parity, expected, 0.01));
    }

    #[test]
    fn test_greeks() {
        let greeks = call_greeks(100.0, 100.0, 1.0, 0.05, 0.2);
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_binomial() {
        let bs = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
        let bin = binomial_price(100.0, 100.0, 1.0, 0.05, 0.2, 100, true);
        assert!((bs - bin).abs() < 0.5); // Should converge
    }

    #[test]
    fn test_portfolio() {
        let returns = vec![0.10, 0.05, -0.02];
        let weights = vec![0.5, 0.3, 0.2];
        let port_return = portfolio_return(&returns, &weights);
        assert!(approx_eq(port_return, 0.054, 1e-10));
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 90.0, 95.0, 80.0, 85.0];
        let dd = max_drawdown(&prices);
        assert!(approx_eq(dd, (110.0 - 80.0) / 110.0, 1e-10));
    }

    #[test]
    fn test_bond_price() {
        let price = bond_price(1000.0, 0.05, 0.05, 10);
        // At par when coupon = yield
        assert!(approx_eq(price, 1000.0, 1.0));
    }

    #[test]
    fn test_cagr() {
        let r = cagr(100.0, 200.0, 10.0);
        assert!(approx_eq(r, 0.0718, 0.001));
    }
}
