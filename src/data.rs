use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Int, Tensor, TensorData, Shape},
};
use image::{DynamicImage, ImageReader};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

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

    pub fn from_index(index: usize) -> Self {
        match index {
            0 => MalariaLabel::Parasitized,
            1 => MalariaLabel::Uninfected,
            _ => panic!("Invalid index for MalariaLabel"),
        }
    }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        let data: Vec<i64> = vec![self.to_index() as i64];
        Tensor::from_data(TensorData::new(data, Shape::new([1])), device)
    }
}

#[derive(Debug, Clone)]
pub struct MalariaItem {
    pub image: DynamicImage,
    pub label: MalariaLabel,
}

impl MalariaItem {
    pub fn new(image_path: PathBuf, label: MalariaLabel) -> anyhow::Result<Self> {
        let image = ImageReader::open(image_path)?.decode()?;
        Ok(Self { image, label })
    }
}

pub struct MalariaBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MalariaBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, MalariaItem, MalariaBatch<B>> for MalariaBatcher<B> {
    fn batch(&self, items: Vec<MalariaItem>, device: &B::Device) -> MalariaBatch<B> {
        let batch_size = items.len();
        let height = 128;
        let width = 128;
        
        let mut images = Vec::with_capacity(batch_size * 3 * height * width);
        let mut labels = Vec::with_capacity(batch_size);
        
        for item in items {
            // Redimensionner l'image
            let img = item.image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::Lanczos3,
            );
            
            // Convertir en RGB et normaliser
            let rgb_img = img.to_rgb8();
            
            for c in 0..3 {
                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x as u32, y as u32);
                        let value = pixel[c] as f32 / 255.0;
                        images.push(value);
                    }
                }
            }
            
            labels.push(item.label.to_index() as i64);
        }
        
        let images_data = TensorData::new(images, Shape::new([batch_size, 3, height, width]));
        let images_tensor = Tensor::from_data(images_data, device);
        
        let targets_data = TensorData::new(labels, Shape::new([batch_size]));
        let targets_tensor: Tensor<B, 1, Int> = Tensor::from_data(targets_data, device);
        
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

pub struct MalariaDataset {
    pub items: Vec<(PathBuf, MalariaLabel)>,
}

impl MalariaDataset {
    pub fn new(data_dir: &Path) -> anyhow::Result<Self> {
        let mut items = Vec::new();
        
        // Charger les images parasitées
        let parasitized_dir = data_dir.join("Parasitized");
        if parasitized_dir.exists() {
            for entry in std::fs::read_dir(parasitized_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| {
                    ext == "png" || ext == "jpg" || ext == "jpeg"
                }) {
                    items.push((path, MalariaLabel::Parasitized));
                }
            }
        }
        
        // Charger les images non infectées
        let uninfected_dir = data_dir.join("Uninfected");
        if uninfected_dir.exists() {
            for entry in std::fs::read_dir(uninfected_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().map_or(false, |ext| {
                    ext == "png" || ext == "jpg" || ext == "jpeg"
                }) {
                    items.push((path, MalariaLabel::Uninfected));
                }
            }
        }
        
        Ok(Self { items })
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    pub fn split(&self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let mut rng = rand::rng();
        let mut shuffled: Vec<_> = self.items.clone();
        shuffled.shuffle(&mut rng);
        
        let total = shuffled.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;
        
        let train_items = shuffled[..train_end].to_vec();
        let val_items = shuffled[train_end..val_end].to_vec();
        let test_items = shuffled[val_end..].to_vec();
        
        (
            Self { items: train_items },
            Self { items: val_items },
            Self { items: test_items },
        )
    }
    
    pub fn get_batch(&self, indices: &[usize]) -> Vec<MalariaItem> {
        indices
            .iter()
            .filter_map(|&idx| {
                let (path, label) = &self.items[idx];
                MalariaItem::new(path.clone(), label.clone()).ok()
            })
            .collect()
    }
}

pub struct MalariaDataLoader {
    pub dataset: Arc<MalariaDataset>,
    batch_size: usize,
    shuffle: bool,
}

impl MalariaDataLoader {
    pub fn new(dataset: MalariaDataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size,
            shuffle,
        }
    }
    
    pub fn iter(&self) -> MalariaDataLoaderIter {
        let indices: Vec<usize> = (0..self.dataset.len()).collect();
        MalariaDataLoaderIter {
            dataset: Arc::clone(&self.dataset),
            indices,
            batch_size: self.batch_size,
            current: 0,
            shuffle: self.shuffle,
        }
    }
    
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub struct MalariaDataLoaderIter {
    dataset: Arc<MalariaDataset>,
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
    shuffle: bool,
}

impl Iterator for MalariaDataLoaderIter {
    type Item = Vec<usize>;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.len() {
            return None;
        }
        
        if self.shuffle && self.current == 0 {
            let mut rng = rand::rng();
            self.indices.shuffle(&mut rng);
        }
        
        let end = std::cmp::min(self.current + self.batch_size, self.dataset.len());
        let batch_indices = self.indices[self.current..end].to_vec();
        self.current = end;
        
        Some(batch_indices)
    }
}