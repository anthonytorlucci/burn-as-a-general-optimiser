//! Synthetic data generation for the curve fit.
//!
//! [`Dataset::sample`] draws noisy observations from a known [`ExpDecayParams`]
//! so the fit can be checked against ground truth ([`GROUND_TRUTH`]).

use rand_distr::{Distribution, Normal};

use crate::model::ExpDecayParams;

/// Number of synthetic observations generated for the fit.
pub const NSAMPLES: usize = 1000;

/// The parameters the synthetic data is generated from; the fit should recover
/// these.
pub const GROUND_TRUTH: ExpDecayParams = ExpDecayParams {
    a: 0.7,
    k: 0.01,
    b: 0.2,
};

/// Synthetic observations sampled at integer `x = 0..NSAMPLES`.
pub struct Dataset {
    /// Sample positions, i.e. `x[i] == i`.
    pub x: [f32; NSAMPLES],
    /// Noisy observations `y[i] = params.eval(x[i]) + noise`.
    pub y: [f32; NSAMPLES],
}

impl Dataset {
    /// Generates `NSAMPLES` observations from `params` with additive zero-mean
    /// Gaussian noise of standard deviation `noise_std_dev`.
    ///
    /// # Panics
    ///
    /// Panics if `noise_std_dev` is not finite (NaN or infinite).
    pub fn sample(params: ExpDecayParams, noise_std_dev: f64) -> Self {
        let mut rng = rand::rng();
        let normal =
            Normal::new(0.0, noise_std_dev).expect("noise std dev must be finite and non-negative");

        let x = std::array::from_fn(|i| i as f32);
        let y = std::array::from_fn(|i| params.eval(i as f32) + normal.sample(&mut rng) as f32);

        Self { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_axis_is_the_sample_index() {
        let data = Dataset::sample(GROUND_TRUTH, 0.0);
        assert_eq!(data.x.len(), NSAMPLES);
        assert_eq!(data.x[0], 0.0);
        assert_eq!(data.x[NSAMPLES - 1], (NSAMPLES - 1) as f32);
    }

    #[test]
    fn zero_noise_reproduces_the_curve_exactly() {
        // With std dev 0 the Gaussian collapses to its mean (0), so every
        // observation equals the noise-free curve.
        let data = Dataset::sample(GROUND_TRUTH, 0.0);
        for (x, y) in data.x.iter().zip(data.y) {
            assert!(
                (y - GROUND_TRUTH.eval(*x)).abs() < 1e-6,
                "mismatch at x = {x}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "noise std dev")]
    fn non_finite_noise_panics() {
        let _ = Dataset::sample(GROUND_TRUTH, f64::NAN);
    }
}
