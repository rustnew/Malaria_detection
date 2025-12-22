use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
};

#[derive(Config)]
pub struct MalariaModelConfig {
    #[config(default = 0.5)]
    dropout: f64,
    num_classes: usize,
}

impl MalariaModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MalariaModel<B> {
        MalariaModel {
            conv1: Conv2dConfig::new([3, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            conv4: Conv2dConfig::new([128, 256], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            flatten: burn::nn::Flatten::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
            fc1: LinearConfig::new(256 * 8 * 8, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),
            fc3: LinearConfig::new(256, self.num_classes).init(device),
            relu: Relu::new(),
        }
    }

    pub fn init_with<B: AutodiffBackend>(&self, device: &B::Device) -> MalariaModel<B> {
        self.init(device)
    }
}

#[derive(Module, Debug)]
pub struct MalariaModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    flatten: burn::nn::Flatten,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> MalariaModel<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Convolutional layers with ReLU activations
        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv3.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv4.forward(x);
        let x = self.relu.forward(x);
        
        // Adaptive pooling
        let x = self.pool.forward(x);
        
        // Flatten
        let x = self.flatten.forward(x);
        
        // Fully connected layers with dropout
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        
        // Output layer
        self.fc3.forward(x)
    }

    pub fn forward_classification(&self, x: Tensor<B, 4>) -> Tensor<B, 1> {
        let output = self.forward(x);
        output.argmax(1)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub shuffle: bool,
    pub device: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            num_epochs: 10,
            batch_size: 32,
            num_workers: 4,
            shuffle: true,
            device: "cpu".to_string(),
        }
    }
}