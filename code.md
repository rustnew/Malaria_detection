use burn::{
    prelude::*,
    tensor::{backend::Backend, Int, Tensor, TensorData, Shape},
};
use image::{DynamicImage, imageops::FilterType};
use rand::seq::SliceRandom;
use rand::rng;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};

// Structure pour images PRÉTRAITÉES avec dimensions
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub data: Vec<f32>,
    pub label: MalariaLabel,
    pub height: usize,
    pub width: usize,
}

impl PreprocessedImage {
    pub fn new(img: &DynamicImage, label: MalariaLabel, size: (usize, usize)) -> Self {
        let (width, height) = (size.0 as u32, size.1 as u32);
        let img = img.resize_exact(width, height, FilterType::Lanczos3);
        let rgb_img = img.to_rgb8();
        
        let mut data = Vec::with_capacity(3 * height as usize * width as usize);
        
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb_img.get_pixel(x, y);
                data.push(pixel[0] as f32 / 255.0);
                data.push(pixel[1] as f32 / 255.0);
                data.push(pixel[2] as f32 / 255.0);
            }
        }
        
        Self { 
            data, 
            label,
            height: height as usize,
            width: width as usize,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MalariaLabel {
    Parasitized,
    Uninfected,
}

impl MalariaLabel {
    pub fn to_index(&self) -> usize {
        match self {
            MalariaLabel::Parasitized => 0,
            MalariaLabel::Uninfected => 1,
        }
    }
}

// Dataset
pub struct MalariaDataset {
    pub items: Vec<(PathBuf, MalariaLabel)>,
    pub preprocessed_cache: Option<Vec<PreprocessedImage>>,
    pub image_size: (usize, usize),
}

impl MalariaDataset {
    // Ajouter une méthode len()
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    // Ajouter une méthode is_empty()
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    // Méthode new() manquante
    pub fn new(data_path: &std::path::Path, image_size: (usize, usize)) -> anyhow::Result<Self> {
        // Implémentation de base - chargez vos images ici
        let mut items = Vec::new();
        
        // Parcourez les dossiers et ajoutez les images
        let parasitized_path = data_path.join("Parasitized");
        let uninfected_path = data_path.join("Uninfected");
        
        if parasitized_path.exists() && parasitized_path.is_dir() {
            for entry in std::fs::read_dir(parasitized_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "png") {
                    items.push((path, MalariaLabel::Parasitized));
                }
            }
        }
        
        if uninfected_path.exists() && uninfected_path.is_dir() {
            for entry in std::fs::read_dir(uninfected_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "png") {
                    items.push((path, MalariaLabel::Uninfected));
                }
            }
        }
        
        Ok(Self {
            items,
            preprocessed_cache: None,
            image_size,
        })
    }
    
    // Méthode preprocess_all() 
    pub fn preprocess_all(&mut self) -> anyhow::Result<()> {
        println!("Prétraitement de {} images...", self.items.len());
        let mut cache = Vec::with_capacity(self.items.len());
        
        for (path, label) in &self.items {
            match image::open(path) {
                Ok(img) => {
                    let preprocessed = PreprocessedImage::new(&img, label.clone(), self.image_size);
                    cache.push(preprocessed);
                }
                Err(e) => {
                    eprintln!("Erreur lors du chargement de {:?}: {}", path, e);
                }
            }
        }
        
        self.preprocessed_cache = Some(cache);
        println!("✅ Prétraitement terminé");
        Ok(())
    }
    
    // Méthode get_batch_preprocessed() pour accès publique
    pub fn get_batch_preprocessed(&self, indices: &[usize]) -> Vec<PreprocessedImage> {
        if let Some(cache) = &self.preprocessed_cache {
            indices.iter()
                .filter_map(|&idx| cache.get(idx))
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    // Méthode split()
    pub fn split(self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let total = self.items.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;
        
        let mut items = self.items;
        let mut rng = rng();
        items.shuffle(&mut rng);
        
        let train_items = items[0..train_end].to_vec();
        let val_items = items[train_end..val_end].to_vec();
        let test_items = items[val_end..].to_vec();
        
        (
            MalariaDataset {
                items: train_items,
                preprocessed_cache: None,
                image_size: self.image_size,
            },
            MalariaDataset {
                items: val_items,
                preprocessed_cache: None,
                image_size: self.image_size,
            },
            MalariaDataset {
                items: test_items,
                preprocessed_cache: None,
                image_size: self.image_size,
            },
        )
    }
}

// Batcher OPTIMISÉ avec buffer réutilisable
pub struct MalariaBatcher<B: Backend> {
    device: B::Device,
    image_buffer: RwLock<Vec<f32>>,
    label_buffer: RwLock<Vec<i64>>,
}

impl<B: Backend> MalariaBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { 
            device,
            image_buffer: RwLock::new(Vec::new()),
            label_buffer: RwLock::new(Vec::new()),
        }
    }
    
    // ⚡ BATcher ULTRA-RAPIDE
    pub fn batch_preprocessed(&self, items: &[PreprocessedImage]) -> MalariaBatch<B> {
        let batch_size = items.len();
        
        if batch_size == 0 {
            return MalariaBatch {
                images: Tensor::zeros([0, 3, 128, 128], &self.device),
                targets: Tensor::zeros([0], &self.device),
            };
        }
        
        // ✅ DIMENSIONS DIRECTES (pas de calcul)
        let height = items[0].height;
        let width = items[0].width;
        
        // ✅ RÉUTILISATION DES BUFFERS
        let mut image_buffer = self.image_buffer.write().unwrap();
        let mut label_buffer = self.label_buffer.write().unwrap();
        
        // Redimensionnement intelligent
        let needed_capacity = batch_size * 3 * height * width;
        if image_buffer.capacity() < needed_capacity {
            image_buffer.reserve(needed_capacity);
        }
        
        if label_buffer.capacity() < batch_size {
            label_buffer.reserve(batch_size);
        }
        
        // Vidage rapide
        image_buffer.clear();
        label_buffer.clear();
        
        // Remplissage
        for item in items {
            image_buffer.extend_from_slice(&item.data);
            label_buffer.push(item.label.to_index() as i64);
        }
        
        // Création des tenseurs
        let images_tensor = Tensor::from_data(
            TensorData::new(image_buffer.clone(), Shape::new([batch_size, 3, height, width])),
            &self.device,
        );
        
        let targets_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(label_buffer.clone(), Shape::new([batch_size])),
            &self.device,
        );
        
        MalariaBatch { 
            images: images_tensor, 
            targets: targets_tensor 
        }
    }
}

#[derive(Debug, Clone)]
pub struct MalariaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

// DataLoader OPTIMISÉ
pub struct MalariaDataLoader {
    dataset: Arc<MalariaDataset>,
    indices: Vec<usize>,
    batch_size: usize,
}

impl MalariaDataLoader {
    pub fn new(dataset: MalariaDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        
        // Shuffle une seule fois
        if shuffle {
            let mut rng = rng();
            indices.shuffle(&mut rng);
        }
        
        Self {
            dataset: Arc::new(dataset),
            indices,
            batch_size,
        }
    }
    
    pub fn iter(&self) -> MalariaDataLoaderIter {
        MalariaDataLoaderIter {
            dataset: Arc::clone(&self.dataset),
            indices: &self.indices,
            batch_size: self.batch_size,
            current: 0,
        }
    }
    
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
    
    // Méthode publique pour accéder aux données prétraitées
    pub fn get_batch_preprocessed(&self, indices: &[usize]) -> Vec<PreprocessedImage> {
        self.dataset.get_batch_preprocessed(indices)
    }
}

pub struct MalariaDataLoaderIter<'a> {
    dataset: Arc<MalariaDataset>,
    indices: &'a [usize],
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for MalariaDataLoaderIter<'a> {
    type Item = &'a [usize];
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }
        
        let end = std::cmp::min(self.current + self.batch_size, self.indices.len());
        let batch_slice = &self.indices[self.current..end];
        self.current = end;
        
        Some(batch_slice)
    }
}





mod data;
mod metrics;
mod model;
mod trainer;

use anyhow::Result;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};

use crate::{
    data::MalariaDataset,
    model::{MalariaModelConfig, TrainingConfig},
    trainer::MalariaTrainer,
};

fn main() -> Result<()> {
    println!("========================================");
    println!("   CLASSIFICATION DE FROTTIS SANGUINS   ");
    println!("========================================\n");
    
    // Configuration
    let device = NdArrayDevice::Cpu;
    let image_size = (128, 128);
    
    // 1. CHARGEMENT DU DATASET
    println!("📥 ÉTAPE 1: Chargement du dataset...");
    let dataset = MalariaDataset::new(std::path::Path::new("data"), image_size)?;
    
    if dataset.is_empty() {
        eprintln!("❌ Aucune image trouvée!");
        return Ok(());
    }
    
    println!("   ✅ {} images trouvées", dataset.len());
    
    // 2. SPLIT
    let (train_dataset, val_dataset, test_dataset) = dataset.split(0.7, 0.15);
    println!("   ✅ Split: {} train, {} val, {} test", 
             train_dataset.len(), val_dataset.len(), test_dataset.len());
    
    // 3. CONFIGURATION DU MODÈLE
    let model_config = MalariaModelConfig::new(2);
    let train_config = TrainingConfig {
        learning_rate: 1e-3,
        num_epochs: 5,  // Réduit pour le test
        batch_size: 16,  // Réduit pour le test
        num_workers: 1,
        shuffle: true,
        device: "cpu".to_string(),
    };
    
    // 4. ENTRAÎNEMENT
    println!("\n🚀 ÉTAPE 2: Démarrage de l'entraînement...");
    
    type Backend = Autodiff<NdArray>;
    let mut trainer = MalariaTrainer::<Backend>::new(
        model_config,
        train_config,
        device,
    );
    
    let _metrics_tracker = trainer.train(train_dataset, val_dataset)?;
    
    // 5. TEST FINAL
    println!("\n🧪 ÉTAPE 3: Test final...");
    test_model(&trainer, test_dataset)?;
    
    println!("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!");
    
    Ok(())
}

fn test_model<B: burn::tensor::backend::AutodiffBackend>(
    trainer: &MalariaTrainer<B>,
    mut test_dataset: MalariaDataset,
) -> Result<()> {
    // Préprocesser le test set
    test_dataset.preprocess_all()?;
    
    let test_loader = crate::data::MalariaDataLoader::new(
        test_dataset,
        16,
        false,
    );
    
    let (test_loss, test_acc, _) = trainer.validate(&test_loader);
    
    println!("\n📊 RÉSULTATS FINAUX:");
    println!("   Précision Test: {:.2}%", test_acc * 100.0);
    println!("   Perte Test: {:.4}", test_loss);
    
    Ok(())
}


use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: [[usize; 2]; 2],
}

impl ClassificationMetrics {
    pub fn new() -> Self {
        Self {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            confusion_matrix: [[0, 0], [0, 0]],
        }
    }
    
    pub fn calculate(&mut self, predictions: &[usize], targets: &[usize]) {
        let mut tp = 0; // True Positives (Parasitized correctly classified)
        let mut tn = 0; // True Negatives (Uninfected correctly classified)
        let mut fp = 0; // False Positives (Uninfected classified as Parasitized)
        let mut fn_val = 0; // False Negatives (Parasitized classified as Uninfected)
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            match (*pred, *target) {
                (0, 0) => tp += 1,  // Correctly identified as Parasitized
                (1, 1) => tn += 1,  // Correctly identified as Uninfected
                (0, 1) => fp += 1,  // Uninfected classified as Parasitized
                (1, 0) => fn_val += 1, // Parasitized classified as Uninfected
                _ => {}
            }
        }
        
        self.confusion_matrix = [[tp, fn_val], [fp, tn]];
        
        let total = predictions.len() as f64;
        self.accuracy = (tp + tn) as f64 / total;
        
        self.precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        
        self.recall = if tp + fn_val > 0 {
            tp as f64 / (tp + fn_val) as f64
        } else {
            0.0
        };
        
        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        } else {
            0.0
        };
    }
    
    pub fn print_summary(&self) {
        println!("=== Classification Metrics ===");
        println!("Accuracy:  {:.4}", self.accuracy);
        println!("Precision: {:.4}", self.precision);
        println!("Recall:    {:.4}", self.recall);
        println!("F1-Score:  {:.4}", self.f1_score);
        println!("Confusion Matrix:");
        println!("                Predicted");
        println!("                P     U");
        println!("Actual  P     {:4}  {:4}", self.confusion_matrix[0][0], self.confusion_matrix[0][1]);
        println!("        U     {:4}  {:4}", self.confusion_matrix[1][0], self.confusion_matrix[1][1]);
        println!("==============================");
    }
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: f64,
    pub train_accuracy: f64,
    pub val_accuracy: f64,
    pub learning_rate: f64,
}

pub struct MetricsTracker {
    history: VecDeque<TrainingMetrics>,
    max_history: usize,
}

impl MetricsTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }
    
    pub fn add(&mut self, metrics: TrainingMetrics) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }
    
    pub fn get_best_epoch(&self) -> Option<&TrainingMetrics> {
        self.history.iter().max_by(|a, b| {
            a.val_accuracy.partial_cmp(&b.val_accuracy).unwrap()
        })
    }
    
    pub fn print_history(&self) {
        println!("┌─────────┬────────────┬────────────┬──────────────┬──────────────┬────────────┐");
        println!("│ Epoch   │ Train Loss │ Val Loss   │ Train Acc    │ Val Acc      │ LR         │");
        println!("├─────────┼────────────┼────────────┼──────────────┼──────────────┼────────────┤");
        
        for metrics in &self.history {
            println!("│ {:7} │ {:10.4} │ {:10.4} │ {:12.4} │ {:12.4} │ {:10.6} │",
                metrics.epoch,
                metrics.train_loss,
                metrics.val_loss,
                metrics.train_accuracy,
                metrics.val_accuracy,
                metrics.learning_rate
            );
        }
        println!("└─────────┴────────────┴────────────┴──────────────┴──────────────┴────────────┘");
        
        if let Some(best) = self.get_best_epoch() {
            println!("Best epoch: {} with validation accuracy: {:.4}%", 
                best.epoch, best.val_accuracy * 100.0);
        }
    }
}



// model.rs - CORRIGÉ

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
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

    pub fn init_with<B: AutodiffBackend>(&self, device: &B::Device) -> MalariaModel<B> {
        self.init(device)
    }
}

// CORRECTION: Retirer Clone du derive
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
        
        // Flatten manually
        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);
        
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

    pub fn forward_classification(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let output = self.forward(x);
        output
    }
}

#[derive(Config, Debug)]
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


[package]
name = "Burn_model"
version = "0.1.0"
edition = "2021"

[dependencies]
burn = { version = "0.19.1", features = ["std", "train", "ndarray", "autodiff"], default-features = false }
burn-ndarray = "0.19.1"
image = "0.25.9"
rand = "0.9"
rayon = "1.10"  # NOUVEAU: Parallélisation
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
env_logger = "0.11"
chrono = "0.4"
burn-train = "0.19.1"
parking_lot = "0.12.5"
tokio = "1.48.0"
log = "0.4.29"
thiserror = "2.0.17"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = true
