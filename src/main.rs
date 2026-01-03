pub mod data;
pub mod model;
pub mod metrics;
pub mod trainer;

use burn::{
    backend::{Autodiff, NdArray},
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::{Backend, AutodiffBackend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
    nn::loss::CrossEntropyLoss,
};
use burn_ndarray::NdArrayDevice;

// Types de batchs
type TrainBatch<B> = data::MalariaBatch<B>;
type ValidBatch<B> = data::MalariaBatch<B>;

// Implémentation des traits REQUIS pour Burn
impl<B: AutodiffBackend> burn::train::TrainStep<TrainBatch<B>, burn::train::ClassificationOutput<B>> for model::MalariaModel<B> {
    fn step(&self, batch: TrainBatch<B>) -> burn::train::TrainOutput<burn::train::ClassificationOutput<B>> {
        let output = self.forward(batch.images.clone());
        let loss = CrossEntropyLoss::new(None, &batch.images.device())
            .forward(output.clone(), batch.targets.clone());
        
        let item = burn::train::ClassificationOutput::new(loss.clone(), output, batch.targets);
        burn::train::TrainOutput::new(self, loss.backward(), item)
    }
}

impl<B: Backend> burn::train::ValidStep<ValidBatch<B>, burn::train::ClassificationOutput<B>> for model::MalariaModel<B> {
    fn step(&self, batch: ValidBatch<B>) -> burn::train::ClassificationOutput<B> {
        let output = self.forward(batch.images);
        let loss = CrossEntropyLoss::new(None, &output.device())
            .forward(output.clone(), batch.targets.clone());
        
        burn::train::ClassificationOutput::new(loss, output, batch.targets)
    }
}

fn main() -> anyhow::Result<()> {
    println!("========================================");
    println!("   CLASSIFICATION DE FROTTIS SANGUINS   ");
    println!("========================================\n");
    
    // Configuration
    let device = NdArrayDevice::Cpu;
    let image_size = (128, 128);
    
    // 1. CHARGEMENT DU DATASET
    println!("📥 ÉTAPE 1: Chargement du dataset...");
    let dataset = match data::MalariaDataset::new(std::path::Path::new("data"), image_size) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("❌ Erreur chargement dataset: {}", e);
            eprintln!("ℹ️  Structure attendue:");
            eprintln!("   data/");
            eprintln!("   ├── Parasitized/");
            eprintln!("   │   └── *.png");
            eprintln!("   └── Uninfected/");
            eprintln!("       └── *.png");
            return Ok(());
        }
    };
    
    if dataset.is_empty() {
        eprintln!("❌ Aucune image trouvée dans data/");
        return Ok(());
    }
    
    println!("   ✅ {} images trouvées", dataset.len());
    
    // 2. SPLIT DES DONNÉES
    let (train_dataset, val_dataset, _test_dataset) = dataset.split(0.7, 0.15);
    
    println!("\n📊 Répartition des données:");
    println!("   Train: {} images", train_dataset.len());
    println!("   Validation: {} images", val_dataset.len());
    println!("   Test: {} images", _test_dataset.len());
    
    // 3. CONFIGURATIONS
    let model_config = model::MalariaModelConfig::create(2);
    let optim_config = AdamConfig::new();
    
    // Paramètres d'entraînement
    let num_epochs = 5;
    let batch_size = 16;
    let learning_rate = 1e-3;
    
    println!("\n⚙️  Configuration:");
    println!("   Epochs: {}", num_epochs);
    println!("   Batch size: {}", batch_size);
    println!("   Learning rate: {}", learning_rate);
    
    // 4. CRÉATION DES DATALOADERS
    println!("\n📊 ÉTAPE 2: Préparation des dataloaders...");
    
    // CORRECTION: Utiliser NdArray pour la validation, Autodiff<NdArray> pour l'entraînement
    type TrainBackend = Autodiff<NdArray>;
    type ValidBackend = NdArray;
    
    // Batcher pour l'entraînement (avec autodiff)
    let batcher_train = data::MalariaBatcher::<TrainBackend>::new(device.clone());
    
    // Batcher pour la validation (sans autodiff)
    let batcher_valid = data::MalariaBatcher::<ValidBackend>::new(device.clone());
    
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(1)
        .build(train_dataset);
    
    let dataloader_val = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .shuffle(0)
        .num_workers(1)
        .build(val_dataset);
    
    println!("    Dataloaders créés");
    
    // 5. INITIALISATION DU MODÈLE
    let model = model_config.init_with::<TrainBackend>(&device);
    
    // 6. CRÉATION ET ENTRAÎNEMENT DU LEARNER
    println!("\n🚀 ÉTAPE 3: Démarrage de l'entraînement...");
    
    let artifact_dir = "checkpoints/";
    std::fs::create_dir_all(artifact_dir)?;
    
    let start_time = std::time::Instant::now();
    
    // Créer le learner avec le bon type
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(num_epochs)
        .build(model, optim_config.init(), learning_rate);
    
    
    let _model_trained = learner.fit(dataloader_train, dataloader_val);
    
    let training_duration = start_time.elapsed();
    println!("\n✅ Entraînement terminé en {:.2?}", training_duration);
    
    println!("💾 Les checkpoints sont sauvegardés dans: {}", artifact_dir);
    
    Ok(())
}