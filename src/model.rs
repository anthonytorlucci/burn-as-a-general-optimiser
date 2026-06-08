//! The model being fitted: an exponential-decay curve `y = a·e^(-k·x) + b`.
//!
//! Two representations live here:
//!
//! - [`ExpDecayParams`] — plain `f32` parameters with a CPU [`eval`](ExpDecayParams::eval),
//!   used to generate synthetic data and to report the recovered fit. It is
//!   backend-agnostic and carries no Burn machinery.
//! - [`CustomModel`] — the *trainable* version, whose parameters are wrapped in
//!   [`Param`] so Burn's optimizer will update them via autodiff.

use burn::module::{Module, Param};
use burn::tensor::ElementConversion;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Parameters of the exponential-decay curve `y = a·e^(-k·x) + b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpDecayParams {
    /// Initial amplitude.
    pub a: f32,
    /// Decay rate (expected to be small).
    pub k: f32,
    /// Vertical offset / asymptote.
    pub b: f32,
}

impl ExpDecayParams {
    /// Evaluates the curve at a single point `x` on the CPU.
    ///
    /// This is independent of any Burn backend and is used for synthetic data
    /// generation and validation, not for the gradient-based fit itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use burn_as_a_general_optimiser::ExpDecayParams;
    ///
    /// // With k = 0 there is no decay, so the curve is the constant a + b.
    /// let params = ExpDecayParams { a: 1.0, k: 0.0, b: 2.0 };
    /// assert_eq!(params.eval(100.0), 3.0);
    /// ```
    pub fn eval(&self, x: f32) -> f32 {
        self.a * (-self.k * x).exp() + self.b
    }
}

/// Trainable Burn module fitting [`ExpDecayParams`] by gradient descent.
///
/// Each scalar parameter is wrapped in a [`Param`] so it becomes a value Burn's
/// optimizer will update. [`forward`](Self::forward) computes the curve with
/// tensor ops so gradients flow through it.
#[derive(Module, Debug)]
pub struct CustomModel<B: Backend> {
    /// Trainable amplitude `a`.
    pub param_a: Param<Tensor<B, 1>>,
    /// Trainable decay rate `k`.
    pub param_k: Param<Tensor<B, 1>>,
    /// Trainable offset `b`.
    pub param_b: Param<Tensor<B, 1>>,
}

impl<B: Backend> CustomModel<B> {
    /// Initialises the model with parameters drawn from plausible priors.
    ///
    /// The ranges bias the search toward the expected regime (small `k`, an
    /// amplitude near 1, a small offset) rather than starting from zero.
    pub fn new(device: &B::Device) -> Self {
        CustomModel {
            param_a: Param::from_tensor(Tensor::random(
                [1],
                burn::tensor::Distribution::Uniform(0.5, 1.0),
                device,
            )),
            param_k: Param::from_tensor(Tensor::random(
                [1],
                // We expect this value to be small.
                burn::tensor::Distribution::Uniform(0.0001, 0.1),
                device,
            )),
            param_b: Param::from_tensor(Tensor::random(
                [1],
                burn::tensor::Distribution::Uniform(0.1, 0.3),
                device,
            )),
        }
    }

    /// Computes `y = a·e^(-k·x) + b` for every entry of `x`, with broadcasting.
    pub fn forward(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let a = self.param_a.val();
        let k = self.param_k.val();
        let b = self.param_b.val();

        // Gradients through `exp()` can explode, so the caller keeps the
        // learning rate low (Burn has no clip on `GradientsParams`).
        let exp_neg_k_x = k.neg().mul(x).exp();
        a.mul(exp_neg_k_x).add(b)
    }

    /// Reads the current parameter estimates back out as plain `f32` scalars.
    pub fn params(&self) -> ExpDecayParams {
        ExpDecayParams {
            a: self.param_a.val().into_scalar().elem::<f32>(),
            k: self.param_k.val().into_scalar().elem::<f32>(),
            b: self.param_b.val().into_scalar().elem::<f32>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;
    use burn::backend::flex::FlexDevice;

    type TestBackend = Flex<f32>;

    #[test]
    fn eval_at_zero_is_amplitude_plus_offset() {
        let params = ExpDecayParams {
            a: 0.7,
            k: 0.01,
            b: 0.2,
        };
        // e^0 == 1, so eval(0) == a + b.
        assert_eq!(params.eval(0.0), params.a + params.b);
    }

    #[test]
    fn eval_decays_toward_offset() {
        let params = ExpDecayParams {
            a: 1.0,
            k: 0.5,
            b: 0.2,
        };
        // Larger x decays the amplitude term toward 0, approaching b.
        assert!(params.eval(0.0) > params.eval(1.0));
        assert!(params.eval(1.0) > params.eval(100.0));
        assert!((params.eval(100.0) - params.b).abs() < 1e-6);
    }

    /// Build a model with fixed (non-random) parameters for deterministic tests.
    fn model_with(device: &FlexDevice, a: f32, k: f32, b: f32) -> CustomModel<TestBackend> {
        CustomModel {
            param_a: Param::from_tensor(Tensor::from_floats([a], device)),
            param_k: Param::from_tensor(Tensor::from_floats([k], device)),
            param_b: Param::from_tensor(Tensor::from_floats([b], device)),
        }
    }

    #[test]
    fn forward_matches_cpu_eval() {
        let device = Default::default();
        let truth = ExpDecayParams {
            a: 0.7,
            k: 0.01,
            b: 0.2,
        };
        let model = model_with(&device, truth.a, truth.k, truth.b);

        let xs = [0.0f32, 1.0, 10.0, 100.0];
        let y_hat = model.forward(Tensor::<TestBackend, 1>::from_floats(xs, &device));
        let y_hat: Vec<f32> = y_hat.into_data().to_vec().unwrap();

        for (x, got) in xs.iter().zip(y_hat) {
            assert!((got - truth.eval(*x)).abs() < 1e-6, "mismatch at x = {x}");
        }
    }

    #[test]
    fn params_round_trips_through_the_module() {
        let device = Default::default();
        let model = model_with(&device, 0.7, 0.01, 0.2);

        let recovered = model.params();
        assert!((recovered.a - 0.7).abs() < 1e-6);
        assert!((recovered.k - 0.01).abs() < 1e-6);
        assert!((recovered.b - 0.2).abs() < 1e-6);
    }
}
