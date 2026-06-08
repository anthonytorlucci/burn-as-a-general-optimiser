//! Burn as a general-purpose optimiser.
//!
//! This crate explores using [Burn](https://burn.dev)'s autodiff and optimizers
//! to fit arbitrary parametric models rather than to train conventional neural
//! networks. The worked example fits the three parameters of an exponential
//! decay curve `y = a·e^(-k·x) + b` to noisy synthetic data via gradient
//! descent.
//!
//! The pieces:
//!
//! - [`model`] — the curve, in both plain ([`ExpDecayParams`]) and trainable
//!   ([`CustomModel`]) form.
//! - [`data`] — synthetic, noisy observations to fit against.
//! - [`training`] — the manual `forward` → `backward` → `step` loop.
//!
//! The backend is generic throughout; the concrete backend is chosen only in
//! the binary (`src/main.rs`).

pub mod data;
pub mod model;
pub mod training;

pub use data::{Dataset, GROUND_TRUTH, NSAMPLES};
pub use model::{CustomModel, ExpDecayParams};
pub use training::{TrainConfig, train_static_data};
