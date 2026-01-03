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