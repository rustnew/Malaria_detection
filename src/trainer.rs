use burn::{
    data::{dataloader::batcher::Batcher, dataloader::Dataset},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
    nn::loss::CrossEntropyLoss,
};

use crate::{
    data::{MalariaBatcher, MalariaDataset},
    model::{MalariaModel, MalariaModelConfig},
};

// Configuration pour l'entraînement personnalisé
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub shuffle: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 10,
            batch_size: 32,
            learning_rate: 1e-3,
            shuffle: true,
        }
    }
}

// Structure pour tracker les métriques
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MetricsTracker {
    pub metrics: Vec<TrainingMetrics>,
    pub num_epochs: usize,
}

impl MetricsTracker {
    pub fn new(num_epochs: usize) -> Self {
        Self {
            metrics: Vec::new(),
            num_epochs,
        }
    }
    
    pub fn add(&mut self, metrics: TrainingMetrics) {
        self.metrics.push(metrics);
    }
}

pub struct MalariaTrainer<B: AutodiffBackend> {
    pub model: MalariaModel<B>,
    pub optimizer_config: AdamConfig,
    pub batcher: MalariaBatcher<B>,
    pub loss_fn: CrossEntropyLoss<B>,
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
        let batcher = MalariaBatcher::new(device.clone());
        
        // Configuration de l'optimizer
        let optimizer_config = AdamConfig::new()
            .with_epsilon(1e-8);
        
        let loss_fn = CrossEntropyLoss::new(None, &device);
        
        Self {
            model,
            optimizer_config,
            batcher,
            loss_fn,
            device,
            config: train_config,
        }
    }
    
    // ENTRAÎNEMENT OPTIMISÉ avec API Burn
    pub fn train_epoch(
        &mut self,
        dataset: &MalariaDataset,
    ) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        let mut batch_count = 0;
        
        // Création de l'optimizer avec état
        let mut optimizer = self.optimizer_config.init();
        
        // Création des indices de batch
        let indices: Vec<usize> = (0..dataset.len()).collect();
        let batches: Vec<Vec<usize>> = indices
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // ITÉRATION sur les batches
        for batch_indices in batches {
            // Collecter les items pour ce batch - utiliser la méthode du trait Dataset
            let mut items = Vec::new();
            for &idx in &batch_indices {
                if let Some(item) = Dataset::get(dataset, idx) {
                    items.push(item);
                }
            }
            
            if items.is_empty() {
                continue;
            }
            
            // Création du batch avec le Batcher de Burn
            let batch = self.batcher.batch(items, &self.device);
            
            let output = self.model.forward(batch.images.clone());
            
            // Utilisation de la loss de Burn
            let loss = self.loss_fn.forward(
                output.clone(),
                batch.targets.clone(),
            );
            
            // Backward + optimisation avec l'API Burn
            let grads = loss.backward();
            let gradients_params = GradientsParams::from_grads(grads, &self.model);
            
            self.model = optimizer.step(self.config.learning_rate, self.model.clone(), gradients_params);
            
            total_loss += loss.clone().into_scalar().elem::<f64>();
            
            // Calcul précision avec API Burn
            let batch_size = batch_indices.len();
            let predictions = output.argmax(1);
            let targets = batch.targets;
            
            let predictions_reshaped = predictions.reshape([batch_size]);
            let targets_reshaped = targets.reshape([batch_size]);
            
            // Calcul des prédictions correctes
            let predictions_data = predictions_reshaped.into_data();
            let targets_data = targets_reshaped.into_data();
            
            let correct = predictions_data
                .as_slice::<i64>()
                .unwrap()
                .iter()
                .zip(targets_data.as_slice::<i64>().unwrap().iter())
                .filter(|(&pred, &target)| pred == target)
                .count();
            
            total_correct += correct;
            total_samples += batch_size;
            batch_count += 1;
        }
        
        let avg_loss = if batch_count > 0 {
            total_loss / batch_count as f64
        } else {
            0.0
        };
        
        let accuracy = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };
        
        (avg_loss, accuracy)
    }
    
    // VALIDATION avec API Burn
    pub fn validate(
        &self,
        dataset: &MalariaDataset,
    ) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        let mut batch_count = 0;
        
        let model = &self.model;
        
        // Création des indices de batch pour validation
        let indices: Vec<usize> = (0..dataset.len()).collect();
        let batches: Vec<Vec<usize>> = indices
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // ITÉRATION sur les batches de validation
        for batch_indices in batches {
            // Collecter les items pour ce batch - utiliser la méthode du trait Dataset
            let mut items = Vec::new();
            for &idx in &batch_indices {
                if let Some(item) = Dataset::get(dataset, idx) {
                    items.push(item);
                }
            }
            
            if items.is_empty() {
                continue;
            }
            
            // Création du batch
            let batch = self.batcher.batch(items, &self.device);
            
            // Forward pass sans backward (pour validation)
            let output = model.forward(batch.images.clone());
            
            // Calcul de la loss (pas de backward)
            let loss = self.loss_fn.forward(
                output.clone(),
                batch.targets.clone(),
            );
            
            total_loss += loss.clone().into_scalar().elem::<f64>();
            
            // Calcul précision
            let batch_size = batch_indices.len();
            let predictions = output.argmax(1);
            let targets = batch.targets;
            
            let predictions_reshaped = predictions.reshape([batch_size]);
            let targets_reshaped = targets.clone().reshape([batch_size]);
            
            // Calcul des prédictions correctes
            let predictions_data = predictions_reshaped.into_data();
            let targets_data = targets_reshaped.into_data();
            
            let correct = predictions_data
                .as_slice::<i64>()
                .unwrap()
                .iter()
                .zip(targets_data.as_slice::<i64>().unwrap().iter())
                .filter(|(&pred, &target)| pred == target)
                .count();
            
            total_correct += correct;
            total_samples += batch_size;
            batch_count += 1;
        }
        
        let avg_loss = if batch_count > 0 {
            total_loss / batch_count as f64
        } else {
            0.0
        };
        
        let accuracy = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };
        
        (avg_loss, accuracy)
    }
    
    // MÉTHODE PRINCIPALE OPTIMISÉE avec API Burn
    pub fn train(
        &mut self,
        train_dataset: crate::data::MalariaDataset,
        val_dataset: crate::data::MalariaDataset,
    ) -> anyhow::Result<MetricsTracker> {
        let mut tracker = MetricsTracker::new(self.config.num_epochs);
        
        println!("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT...");
        println!("📊 Configuration: {} epochs, LR: {}, Batch: {}", 
                 self.config.num_epochs, 
                 self.config.learning_rate,
                 self.config.batch_size);
        
        let start_time = std::time::Instant::now();
        
        for epoch in 0..self.config.num_epochs {
            let epoch_start = std::time::Instant::now();
            
            println!("\n--- ÉPOQUE {}/{} ---", epoch + 1, self.config.num_epochs);
            
            // Entraînement
            let (train_loss, train_acc) = self.train_epoch(&train_dataset);
            
            // Validation
            let (val_loss, val_acc) = self.validate(&val_dataset);
            
            let epoch_duration = epoch_start.elapsed();
            
            println!("   Perte: {:.4} → {:.4}", train_loss, val_loss);
            println!("   Précision: {:.1}% → {:.1}%", train_acc * 100.0, val_acc * 100.0);
            println!("   Temps epoch: {:.2?}", epoch_duration);
            
            // Sauvegarde des métriques
            tracker.add(TrainingMetrics {
                epoch: epoch + 1,
                train_loss,
                val_loss,
                train_accuracy: train_acc,
                val_accuracy: val_acc,
                learning_rate: self.config.learning_rate,
            });
            
            // Sauvegarde conditionnelle
            if (epoch + 1) % 2 == 0 || epoch == self.config.num_epochs - 1 {
                self.save_checkpoint(epoch + 1)?;
            }
        }
        
        let total_duration = start_time.elapsed();
        println!("\n✅ ENTRAÎNEMENT TERMINÉ en {:.2?}", total_duration);
        
        Ok(tracker)
    }
    
    // SAUVEGARDE du modèle
    pub fn save_checkpoint(&self, epoch: usize) -> anyhow::Result<()> {
        let checkpoint_dir = "checkpoints";
        std::fs::create_dir_all(checkpoint_dir)?;
        
        let model_path = format!("{}/epoch_{}.json", checkpoint_dir, epoch);
        
        #[derive(serde::Serialize)]
        struct CheckpointInfo {
            epoch: usize,
            learning_rate: f64,
            timestamp: String,
        }
        
        let info = CheckpointInfo {
            epoch,
            learning_rate: self.config.learning_rate,
            timestamp: chrono::Local::now().to_rfc3339(),
        };
        
        let json = serde_json::to_string_pretty(&info)?;
        std::fs::write(&model_path, json)?;
        
        println!("   ✅ Checkpoint sauvegardé: {}", model_path);
        
        Ok(())
    }
}