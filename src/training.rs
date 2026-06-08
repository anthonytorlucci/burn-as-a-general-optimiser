//! The manual training loop.
//!
//! Follows Burn's [custom training loop] pattern (deliberately *not* the
//! higher-level `Learner` API): `forward` → `MseLoss` → `loss.backward()` →
//! `GradientsParams::from_grads` → `optimizer.step`, rebinding the returned
//! model each iteration since Burn models are immutable.
//!
//! [custom training loop]: https://burn.dev/books/burn/custom-training-loop.html

use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;

use crate::data::{Dataset, GROUND_TRUTH};
use crate::model::{CustomModel, ExpDecayParams};

/// Hyperparameters for [`train_static_data`].
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// Number of gradient-descent iterations.
    pub epochs: usize,
    /// Optimizer step size. Kept low because gradients through `exp()` can
    /// explode and Burn has no built-in gradient clipping.
    pub learning_rate: f64,
    /// Standard deviation of the Gaussian noise added to the synthetic data.
    pub noise_std_dev: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 1000,
            learning_rate: 1e-4,
            noise_std_dev: 0.05,
        }
    }
}

/// Fits [`CustomModel`] to freshly sampled synthetic data and returns the
/// recovered parameters.
///
/// The data is generated from [`GROUND_TRUTH`]; a successful run converges the
/// returned [`ExpDecayParams`] back toward those values.
pub fn train_static_data<B: AutodiffBackend>(
    device: &B::Device,
    config: TrainConfig,
) -> ExpDecayParams {
    let data = Dataset::sample(GROUND_TRUTH, config.noise_std_dev);
    let x: Tensor<B, 1> = Tensor::from_floats(data.x, device);
    let y: Tensor<B, 1> = Tensor::from_floats(data.y, device);

    // Models are immutable, so the loop rebinds `model` from `optimizer.step`.
    let mut model = CustomModel::<B>::new(device);
    let mut optimizer = AdamConfig::new().init();

    for epoch in 1..=config.epochs {
        let y_hat = model.forward(x.clone());
        let loss = MseLoss::new().forward(y_hat, y.clone(), Reduction::Mean);

        println!(
            "[Train - Epoch {epoch}] Loss {:.3}",
            loss.clone().into_scalar(),
        );

        // Gradients for the current backward pass, linked to each parameter.
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(config.learning_rate, model, grads);
    }

    model.params()
}
