use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    tensor::backend::Backend,
    module::Module,
    config::Config,
};

#[derive(Config, Debug)]
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
            dropout: DropoutConfig::new(self.dropout).init(),
            fc1: LinearConfig::new(256 * 8 * 8, 512).init(device),
            fc2: LinearConfig::new(512, 256).init(device),
            fc3: LinearConfig::new(256, self.num_classes).init(device),
            relu: Relu::new(),
        }
    }

    pub fn init_with<B: burn::tensor::backend::AutodiffBackend>(&self, device: &B::Device) -> MalariaModel<B> {
        self.init(device)
    }
}

// Méthode de création
impl MalariaModelConfig {
    pub fn create(num_classes: usize) -> Self {
        Self {
            dropout: 0.5,
            num_classes,
        }
    }
}

// Modèle
#[derive(Module, Debug)]
pub struct MalariaModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

impl<B: Backend> MalariaModel<B> {
    pub fn forward(&self, x: burn::tensor::Tensor<B, 4>) -> burn::tensor::Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv2.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv3.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.conv4.forward(x);
        let x = self.relu.forward(x);
        
        let x = self.pool.forward(x);
        
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        
        let x = self.fc2.forward(x);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        
        self.fc3.forward(x)
    }
}