//! Binary entry point: selects a concrete backend and runs the curve fit.
//!
//! All the reusable machinery lives in the library crate; `main` only wires up
//! the backend stack (`Autodiff<Wgpu<f32, i32>>` on the default WGPU device) and
//! reports how close the recovered parameters are to ground truth.

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};

use burn_as_a_general_optimiser::{GROUND_TRUTH, TrainConfig, train_static_data};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    let fitted = train_static_data::<Autodiff<MyBackend>>(&device, TrainConfig::default());

    // Validate the fit against the parameters the data was generated from.
    println!(
        "a expected: {:.2}, prediction: {}",
        GROUND_TRUTH.a, fitted.a
    );
    println!(
        "k expected: {:.2}, prediction: {}",
        GROUND_TRUTH.k, fitted.k
    );
    println!(
        "b expected: {:.2}, prediction: {}",
        GROUND_TRUTH.b, fitted.b
    );
}
