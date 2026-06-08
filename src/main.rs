use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::{Module, Param};
use burn::nn::loss::MseLoss;
use burn::nn::loss::Reduction;
use burn::optim::AdamConfig;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use rand_distr::{Distribution, Normal};

const NSAMPLES: usize = 1000;

#[derive(Module, Debug)]
pub struct CustomModel<B: Backend> {
    // A custom, trainable tensor is explicitly wrapped in Param
    //---weights: Param<Tensor<B, 1>>,
    pub param_a: Param<Tensor<B, 1>>,
    pub param_k: Param<Tensor<B, 1>>,
    pub param_b: Param<Tensor<B, 1>>,
}

impl<B: Backend> CustomModel<B> {
    pub fn new(device: &B::Device) -> Self {
        // Typically, we would construct a model from a ModelConfig, but this struct
        // is pretty simple.
        CustomModel {
            param_a: Param::from_tensor(Tensor::random(
                [1], // shape
                burn::tensor::Distribution::Uniform(0.5, 1.0),
                device,
            )),
            param_k: Param::from_tensor(Tensor::random(
                [1],                                              // shape
                burn::tensor::Distribution::Uniform(0.0001, 0.1), // we expect this value to be small
                device,
            )),
            param_b: Param::from_tensor(Tensor::random(
                [1], // shape
                burn::tensor::Distribution::Uniform(0.1, 0.3),
                device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let k = self.param_k.val();
        let a = self.param_a.val();
        let b = self.param_b.val();

        // Compute y = a * e^(-k*x) + b with proper broadcasting
        let neg_k_x = k.neg().mul(x);
        // Numerical Stability Check
        // let max_val = neg_k_x.clone().max().into_scalar();
        // if max_val > 10.0 {
        //     // exp(10) is already ~20,000
        //     println!("Warning: argument to exp too large: {:?}", max_val);
        // }
        let exp_neg_k_x = neg_k_x.exp();
        a.mul(exp_neg_k_x).add(b)
    }
}

fn sample_data() -> [f32; NSAMPLES] {
    // a, k, b are the parameters we wish to find
    let a: f32 = 0.7;
    let k: f32 = 0.01;
    let b: f32 = 0.2;

    // Gaussian noise with zero mean and small standard deviation
    let noise_std_dev = 0.05;
    let mut rng = rand::rng();
    let normal_dist = Normal::new(0.0, noise_std_dev).unwrap();

    // Generate samples: y = a * e^(-kx) + b + Gaussian noise
    std::array::from_fn(|i| {
        let x = i as f32;
        a * (-k * x).exp() + b + normal_dist.sample(&mut rng)
    })
}

// reference: https://burn.dev/books/burn/custom-training-loop.html
fn train_static_data<B: AutodiffBackend>(device: &B::Device) {
    let d = sample_data();
    let y: Tensor<B, 1> = Tensor::from_floats(d, device);

    // Create an array from a range using `from_fn`
    let a: [f32; NSAMPLES] = std::array::from_fn(|i| i as f32);
    let x: Tensor<B, 1> = Tensor::from_floats(a, device);

    // Create the model and the optimizer
    let mut custom_model: CustomModel<B> = CustomModel::new(device);
    let config_optimizer = AdamConfig::new();
    let mut optimizer = config_optimizer.init();

    for epoch in 1..1000 {
        let y_hat = custom_model.forward(x.clone());
        let loss = MseLoss::new().forward(y_hat, y.clone(), Reduction::Mean);

        println!(
            "[Train - Epoch {}] Loss {:.3}",
            epoch,
            loss.clone().into_scalar(),
        );

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &custom_model);
        // Gradients through exponentials can explode; clip them
        // TODO: no clip method in burn !!!
        // Update the model using the optimizer. Use a learning rate of 1e-3, but this a parameter to test
        custom_model = optimizer.step(1e-4, custom_model, grads);
    }

    // Validate model accuracy with known inputs
    println!(
        "a expected: 0.70, prediction: {}",
        custom_model.param_a.val().into_scalar()
    );
    println!(
        "k expected: 0.01, prediction: {}",
        custom_model.param_k.val().into_scalar()
    );
    println!(
        "b expected: 0.20, prediction: {}",
        custom_model.param_b.val().into_scalar()
    );
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = WgpuDevice::default();

    train_static_data::<Autodiff<MyBackend>>(&device);
}
