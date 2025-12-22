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
    // Initialiser le logger
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    
    println!("=== Classification des Cellules de Malaria avec Burn ===");
    println!("Chargement du dataset...");
    
    // Charger le dataset
    let dataset = MalariaDataset::new(std::path::Path::new("data"))?;
    
    if dataset.is_empty() {
        eprintln!("Erreur : Aucune image trouvée dans le dossier data.");
        eprintln!("Structure attendue :");
        eprintln!("  data/Parasitized/*.png (ou .jpg/.jpeg)");
        eprintln!("  data/Uninfected/*.png (ou .jpg/.jpeg)");
        return Ok(());
    }
    
    println!("Total d'images trouvées : {}", dataset.len());
    
    // Diviser le dataset (70% train, 15% validation, 15% test)
    let (train_dataset, val_dataset, test_dataset) = dataset.split(0.7, 0.15);
    
    println!("Division du dataset :");
    println!("  Entraînement :   {} images", train_dataset.len());
    println!("  Validation : {} images", val_dataset.len());
    println!("  Test :       {} images", test_dataset.len());
    
    // Configuration
    let device = NdArrayDevice::Cpu;
    let model_config = MalariaModelConfig::new(2); // 2 classes : Parasitized et Uninfected
    let train_config = TrainingConfig {
        learning_rate: 1e-3,
        num_epochs: 10,  // Réduit pour les tests
        batch_size: 32,
        num_workers: 4,
        shuffle: true,
        device: "cpu".to_string(),
    };
    
    // Créer le trainer
    type Backend = Autodiff<NdArray>;
    let mut trainer = MalariaTrainer::<Backend>::new(
        model_config,
        train_config.clone(),
        device,
    );
    
    // Entraîner le modèle
    let _metrics_tracker = trainer.train(train_dataset, val_dataset)?;
    
    // Tester le modèle
    println!("\n=== Test ===");
    test_model(&trainer, test_dataset)?;
    
    println!("\n=== Entraînement Terminé ===");
    
    Ok(())
}

fn test_model<B: burn::tensor::backend::AutodiffBackend>(
    trainer: &MalariaTrainer<B>,
    test_dataset: MalariaDataset,
) -> Result<()> {
    
    println!("Test sur {} images...", test_dataset.len());
    
    let test_loader = crate::data::MalariaDataLoader::new(
        test_dataset,
        32,
        false,
    );
    
    let (test_loss, test_acc, test_metrics) = trainer.validate(&test_loader);
    
    println!("\n=== Résultats du Test ===");
    println!("Perte Test :    {:.4}", test_loss);
    println!("Précision Test : {:.2}%", test_acc * 100.0);
    test_metrics.print_summary();
    
    Ok(())
}