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