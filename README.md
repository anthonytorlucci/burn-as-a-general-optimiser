# burn-as-a-general-optimiser

Using [Burn](https://burn.dev) — a Rust deep learning framework — as a
**general-purpose optimiser**, rather than for conventional neural networks.

Burn's autodiff and optimizers don't care whether the thing you're fitting is a
deep network or a three-parameter formula. This example fits an exponential
decay curve to noisy data using gradient descent, demonstrating the smallest
useful "model" you can express in Burn: a handful of trainable scalars.

## The example

We fit the parameters `a`, `k`, and `b` of

```
y = a · e^(-k·x) + b
```

to synthetic data generated from known ground-truth values
(`a = 0.7`, `k = 0.01`, `b = 0.2`) plus Gaussian noise. After training, the
recovered parameters should land close to those values.

The trick is that each parameter is a `Param<Tensor<B, 1>>` inside a
`#[derive(Module)]` struct — anything wrapped in `Param` is something Burn's
optimizer will update via backpropagation.

## Running

```bash
cargo run
```

This builds and runs the curve fit, printing the loss each epoch and the final
recovered parameters:

```
[Train - Epoch 1] Loss 0.025
[Train - Epoch 2] Loss 0.025
[Train - Epoch 3] Loss 0.025
...
[Train - Epoch 999] Loss 0.003
a expected: 0.70, prediction: 0.6987837
k expected: 0.01, prediction: 0.011429357
b expected: 0.20, prediction: 0.2207424
```

(Exact predictions vary run to run since the sample data is randomly noised.)

Requires a recent Rust toolchain (the project uses **edition 2024**).

## How it works

`src/main.rs` follows Burn's
[custom training loop](https://burn.dev/books/burn/custom-training-loop.html)
pattern instead of the higher-level `Learner` API:

1. **`CustomModel`** holds the trainable scalars `a`, `k`, `b` and implements
   `forward` using tensor ops so gradients flow through the curve.
2. **`sample_data`** produces the noisy observations from the ground-truth
   parameters.
3. **`train_static_data`** runs the manual loop:
   `forward → MseLoss → loss.backward() → GradientsParams → optimizer.step`,
   reassigning the model each iteration (Burn models are immutable; `step`
   consumes and returns a new one).
4. **`main`** assembles the backend stack: `Autodiff<Wgpu<f32, i32>>` on the
   default WGPU device.

### Notes

- The backend is generic throughout; the concrete backend is chosen only in
  `main`. Autodiff wraps the base backend so `.backward()` exists.
- Gradients through `exp()` can explode, so the learning rate is kept low
  (`1e-4`). Burn currently has no built-in gradient clipping on
  `GradientsParams`.

## Dependencies

- [`burn`](https://crates.io/crates/burn) with `train`, `wgpu`, and `fusion`
  features
- [`rand`](https://crates.io/crates/rand) / [`rand_distr`](https://crates.io/crates/rand_distr)
  for generating the noisy sample data
