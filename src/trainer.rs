use burn::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
    nn::loss::CrossEntropyLoss,
};

use crate::{
    data::{MalariaBatcher, MalariaDataLoader},
    metrics::{ClassificationMetrics, MetricsTracker, TrainingMetrics},
    model::{MalariaModel, MalariaModelConfig, TrainingConfig},
};

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
    
    // ENTRAÎNEMENT OPTIMISÉ
    pub fn train_epoch(
        &mut self,
        train_loader: &MalariaDataLoader,
    ) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        let mut batch_count = 0;
        
        // Création de l'optimizer avec état
        let mut optimizer = self.optimizer_config.init();
        
        // ITÉRATION DIRECTE sans collecter tous les batches
        for batch_indices in train_loader.iter() {
            // Utilisation du cache prétraité via méthode publique
            let items = train_loader.get_batch_preprocessed(batch_indices);
            let batch = self.batcher.batch_preprocessed(&items);
            
            let output = self.model.forward(batch.images.clone());
            
            // Utilisation de la loss pré-créée
            let loss = self.loss_fn.forward(
                output.clone(),
                batch.targets.clone(),
            );
            
            // Backward + optimisation
            let grads = loss.backward();
            let gradients_params = GradientsParams::from_grads(grads, &self.model);
            
            self.model = optimizer.step(self.config.learning_rate, self.model.clone(), gradients_params);
            
            total_loss += loss.clone().into_scalar().elem::<f64>();
            
            // Calcul précision optimisé
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
    
    // VALIDATION OPTIMISÉE (sans autodiff)
    pub fn validate(
        &self,
        val_loader: &MalariaDataLoader,
    ) -> (f64, f64, ClassificationMetrics) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut total_samples = 0;
        let mut batch_count = 0;
        
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        
        // Clone du modèle pour la validation
        let model = &self.model;
        
        // ITÉRATION DIRECTE
        for batch_indices in val_loader.iter() {
            let items = val_loader.get_batch_preprocessed(batch_indices);
            let batch = self.batcher.batch_preprocessed(&items);
            
            // Forward pass sans grad (dans Burn, pas besoin de no_grad explicitement pour juste forward)
            let output = model.forward(batch.images.clone());
            
            // Calcul de la loss (mais pas de backward)
            let loss = self.loss_fn.forward(
                output.clone(),
                batch.targets.clone(),
            );
            
            total_loss += loss.clone().into_scalar().elem::<f64>();
            
            let predictions = output.argmax(1);
            let targets = batch.targets;
            
            let batch_size = batch_indices.len();
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
            
            // Collecter prédictions (uniquement à la fin ou moins fréquemment)
            if all_predictions.len() < 10000 {
                let pred_indices: Vec<usize> = predictions_data
                    .as_slice::<i64>()
                    .unwrap()
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                
                let target_indices: Vec<usize> = targets
                    .into_data()
                    .as_slice::<i64>()
                    .unwrap()
                    .iter()
                    .map(|&x| x as usize)
                    .collect();
                
                all_predictions.extend(pred_indices);
                all_targets.extend(target_indices);
            }
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
        
        let mut metrics = ClassificationMetrics::new();
        metrics.calculate(&all_predictions, &all_targets);
        
        (avg_loss, accuracy, metrics)
    }
    
    // MÉTHODE PRINCIPALE OPTIMISÉE
    pub fn train(
        &mut self,
        mut train_dataset: crate::data::MalariaDataset,
        mut val_dataset: crate::data::MalariaDataset,
    ) -> anyhow::Result<MetricsTracker> {
        // PRÉTRAITEMENT
        println!("\n⚡ PRÉTRAITEMENT DES DONNÉES...");
        train_dataset.preprocess_all()?;
        val_dataset.preprocess_all()?;
        
        // DATALOADERS
        let train_loader = MalariaDataLoader::new(
            train_dataset,
            self.config.batch_size,
            self.config.shuffle,
        );
        
        let val_loader = MalariaDataLoader::new(
            val_dataset,
            self.config.batch_size,
            false,
        );
        
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
            
            let (train_loss, train_acc) = self.train_epoch(&train_loader);
            let (val_loss, val_acc, val_metrics) = self.validate(&val_loader);
            
            let epoch_duration = epoch_start.elapsed();
            
            println!("   Perte: {:.4} → {:.4}", train_loss, val_loss);
            println!("   Précision: {:.1}% → {:.1}%", train_acc * 100.0, val_acc * 100.0);
            println!("   Temps epoch: {:.2?}", epoch_duration);
            
            // Affichage métriques détaillées occasionnellement
            if (epoch + 1) % 5 == 0 {
                println!("   Métriques validation:");
                val_metrics.print_summary();
            }
            
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
    
    // SAUVEGARDE
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