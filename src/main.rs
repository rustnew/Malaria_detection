mod data;
mod metrics;
mod model;
mod trainer;

use anyhow::Result;
use burn::backend::{ndarray::NdArrayDevice, Autodiff, NdArray};
use env_logger::Env;

use crate::{
    data::MalariaDataset,
    model::{MalariaModelConfig, TrainingConfig},
    trainer::MalariaTrainer,
};

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    
    println!("=== Malaria Cell Classification with Burn ===");
    println!("Loading dataset...");
    
    // Load dataset
    let dataset = MalariaDataset::new(std::path::Path::new("data"))?;
    
    if dataset.is_empty() {
        eprintln!("Error: No images found in data directory.");
        eprintln!("Expected structure:");
        eprintln!("  data/Parasitized/*.png (or .jpg/.jpeg)");
        eprintln!("  data/Uninfected/*.png (or .jpg/.jpeg)");
        return Ok(());
    }
    
    println!("Total images found: {}", dataset.len());
    
    // Split dataset (70% train, 15% validation, 15% test)
    let (train_dataset, val_dataset, test_dataset) = dataset.split(0.7, 0.15);
    
    println!("Dataset split:");
    println!("  Training:   {} images", train_dataset.len());
    println!("  Validation: {} images", val_dataset.len());
    println!("  Test:       {} images", test_dataset.len());
    
    // Configuration
    let device = NdArrayDevice::Cpu;
    let model_config = MalariaModelConfig::new(2); // 2 classes: Parasitized and Uninfected
    let train_config = TrainingConfig {
        learning_rate: 1e-3,
        num_epochs: 20,
        batch_size: 32,
        num_workers: 4,
        shuffle: true,
        device: "cpu".to_string(),
    };
    
    // Create trainer
    type Backend = Autodiff<NdArray>;
    let mut trainer = MalariaTrainer::<Backend>::new(
        model_config,
        train_config,
        device,
    );
    
    // Train the model
    let metrics_tracker = trainer.train(train_dataset, val_dataset)?;
    
    // Test the model
    println!("\n=== Testing ===");
    test_model(&trainer, test_dataset)?;
    
    // Save final model
    println!("\nSaving final model...");
    trainer.save_checkpoint(train_config.num_epochs)?;
    
    println!("\n=== Training Complete ===");
    println!("Model saved to checkpoints/");
    
    Ok(())
}

fn test_model<B: burn::tensor::backend::AutodiffBackend>(
    trainer: &MalariaTrainer<B>,
    test_dataset: MalariaDataset,
) -> Result<()> {
    use crate::metrics::ClassificationMetrics;
    
    println!("Testing on {} images...", test_dataset.len());
    
    let test_loader = crate::data::MalariaDataLoader::new(
        test_dataset,
        32,
        false,
    );
    
    let (test_loss, test_acc, test_metrics) = trainer.validate(&test_loader);
    
    println!("\n=== Test Results ===");
    println!("Test Loss:    {:.4}", test_loss);
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);
    test_metrics.print_summary();
    
    Ok(())
}