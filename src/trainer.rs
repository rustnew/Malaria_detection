use burn::{
    data::dataloader::batcher::Batcher,
    optim::{AdamConfig, Optimizer},
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, TrainOutput, TrainStep, ValidStep,
    },
};
use tqdm::tqdm;

use crate::{
    data::{MalariaBatcher, MalariaBatch, MalariaDataLoader, MalariaDataset},
    metrics::{ClassificationMetrics, MetricsTracker, TrainingMetrics},
    model::{MalariaModel, MalariaModelConfig, TrainingConfig},
};

pub struct MalariaTrainer<B: AutodiffBackend> {
    pub model: MalariaModel<B>,
    pub optimizer: burn::optim::Adam<B>,
    pub batcher: MalariaBatcher<B>,
    pub device: B::Device,
    pub config: TrainingConfig,
}

impl<B: AutodiffBackend> MalariaTrainer<B> {
    pub fn new(
        model_config: MalariaModelConfig,
        train_config: TrainingConfig,
        device: B::Device,
    ) -> Self {
        let model = model_config.init_with(&device);
        let optimizer = AdamConfig::new().init(&model);
        let batcher = MalariaBatcher::new(device.clone());
        
        Self {
            model,
            optimizer,
            batcher,
            device,
            config: train_config,
        }
    }
    
    pub fn train_epoch(
        &mut self,
        train_loader: &MalariaDataLoader,
    ) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        
        let batches: Vec<_> = train_loader.iter().collect();
        let progress = tqdm(batches.len() as u64);
        
        for batch_indices in batches {
            let items = train_loader.dataset.get_batch(&batch_indices);
            let batch = self.batcher.batch(items);
            
            let output = self.train_step(batch);
            total_loss += output.loss;
            
            let predictions = output.output.argmax(1);
            let correct = predictions.equal(output.targets).sum().into_scalar().elem::<i32>() as usize;
            
            total_correct += correct;
            total_samples += batch_indices.len();
            
            progress.set_postfix(format!(
                "Loss: {:.4}, Acc: {:.2}%",
                output.loss,
                (correct as f64 / batch_indices.len() as f64) * 100.0
            ));
            progress.inc(1);
        }
        
        progress.close();
        
        let avg_loss = total_loss / batches.len() as f64;
        let accuracy = total_correct as f64 / total_samples as f64;
        
        (avg_loss, accuracy)
    }
    
    pub fn validate(
        &self,
        val_loader: &MalariaDataLoader,
    ) -> (f64, f64, ClassificationMetrics) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        
        let batches: Vec<_> = val_loader.iter().collect();
        
        for batch_indices in batches {
            let items = val_loader.dataset.get_batch(&batch_indices);
            let batch = self.batcher.batch(items);
            
            let output = self.valid_step(batch.clone());
            total_loss += output.loss;
            
            let predictions = output.output.argmax(1);
            let correct = predictions.equal(output.targets).sum().into_scalar().elem::<i32>() as usize;
            
            total_correct += correct;
            total_samples += batch_indices.len();
            
            // Collect predictions and targets for detailed metrics
            let pred_indices: Vec<usize> = predictions
                .into_data()
                .value
                .into_iter()
                .map(|x| x.elem::<usize>())
                .collect();
            
            let target_indices: Vec<usize> = batch.targets
                .into_data()
                .value
                .into_iter()
                .map(|x| x.elem::<usize>())
                .collect();
            
            all_predictions.extend(pred_indices);
            all_targets.extend(target_indices);
        }
        
        let avg_loss = total_loss / batches.len() as f64;
        let accuracy = total_correct as f64 / total_samples as f64;
        
        let mut metrics = ClassificationMetrics::new();
        metrics.calculate(&all_predictions, &all_targets);
        
        (avg_loss, accuracy, metrics)
    }
    
    pub fn train(
        &mut self,
        train_dataset: MalariaDataset,
        val_dataset: MalariaDataset,
    ) -> anyhow::Result<MetricsTracker> {
        let train_loader = MalariaDataLoader::new(
            train_dataset,
            self.config.batch_size,
            self.config.shuffle,
        );
        
        let val_loader = MalariaDataLoader::new(
            val_dataset,
            self.config.batch_size,
            false, // Don't shuffle validation
        );
        
        let mut tracker = MetricsTracker::new(self.config.num_epochs);
        
        println!("Starting training for {} epochs...", self.config.num_epochs);
        println!("Train samples: {}", train_loader.dataset.len());
        println!("Validation samples: {}", val_loader.dataset.len());
        println!("Batch size: {}", self.config.batch_size);
        println!("Learning rate: {}", self.config.learning_rate);
        println!("Device: {:?}", self.device);
        
        for epoch in 0..self.config.num_epochs {
            println!("\nEpoch {}/{}", epoch + 1, self.config.num_epochs);
            println!("{}", "=".repeat(50));
            
            // Training phase
            println!("Training...");
            let (train_loss, train_acc) = self.train_epoch(&train_loader);
            
            // Validation phase
            println!("Validating...");
            let (val_loss, val_acc, val_metrics) = self.validate(&val_loader);
            
            // Print epoch results
            println!("\nEpoch {} Results:", epoch + 1);
            println!("  Train Loss: {:.4}, Train Acc: {:.2}%", 
                train_loss, train_acc * 100.0);
            println!("  Val Loss:   {:.4}, Val Acc:   {:.2}%", 
                val_loss, val_acc * 100.0);
            
            val_metrics.print_summary();
            
            // Track metrics
            tracker.add(TrainingMetrics {
                epoch: epoch + 1,
                train_loss,
                val_loss,
                train_accuracy: train_acc,
                val_accuracy: val_acc,
                learning_rate: self.config.learning_rate,
            });
            
            // Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0 {
                self.save_checkpoint(epoch + 1)?;
            }
        }
        
        println!("\nTraining completed!");
        tracker.print_history();
        
        Ok(tracker)
    }
    
    pub fn save_checkpoint(&self, epoch: usize) -> anyhow::Result<()> {
        let checkpoint_dir = "checkpoints";
        std::fs::create_dir_all(checkpoint_dir)?;
        
        let model_path = format!("{}/model_epoch_{}.bin", checkpoint_dir, epoch);
        let optimizer_path = format!("{}/optimizer_epoch_{}.bin", checkpoint_dir, epoch);
        
        // Save model
        let model_bytes = bincode::serialize(&self.model)?;
        std::fs::write(&model_path, model_bytes)?;
        
        // Save optimizer state
        let optimizer_bytes = bincode::serialize(&self.optimizer)?;
        std::fs::write(&optimizer_path, optimizer_bytes)?;
        
        println!("Checkpoint saved for epoch {}: {}", epoch, model_path);
        
        Ok(())
    }
    
    pub fn load_checkpoint(
        &mut self,
        epoch: usize,
    ) -> anyhow::Result<()> {
        let model_path = format!("checkpoints/model_epoch_{}.bin", epoch);
        let optimizer_path = format!("checkpoints/optimizer_epoch_{}.bin", epoch);
        
        if !std::path::Path::new(&model_path).exists() {
            return Err(anyhow::anyhow!("Checkpoint not found: {}", model_path));
        }
        
        // Load model
        let model_bytes = std::fs::read(&model_path)?;
        self.model = bincode::deserialize(&model_bytes)?;
        
        // Load optimizer
        let optimizer_bytes = std::fs::read(&optimizer_path)?;
        self.optimizer = bincode::deserialize(&optimizer_bytes)?;
        
        println!("Checkpoint loaded from epoch {}", epoch);
        
        Ok(())
    }
}

impl<B: AutodiffBackend> TrainStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaTrainer<B> {
    fn train_step(&mut self, batch: MalariaBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.model.forward(batch.images);
        let loss = burn::nn::loss::CrossEntropyLoss::new(None).forward(
            item.clone(),
            batch.targets.clone(),
        );
        
        let gradients = loss.backward();
        self.optimizer.update(&self.model, gradients);
        
        TrainOutput::new(loss, ClassificationOutput {
            output: item,
            targets: batch.targets,
        })
    }
}

impl<B: AutodiffBackend> ValidStep<MalariaBatch<B>, ClassificationOutput<B>> for MalariaTrainer<B> {
    fn valid_step(&self, batch: MalariaBatch<B>) -> ClassificationOutput<B> {
        let output = self.model.forward(batch.images);
        
        ClassificationOutput {
            output,
            targets: batch.targets,
        }
    }
}