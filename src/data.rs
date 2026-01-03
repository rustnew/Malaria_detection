use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Shape, Tensor, TensorData},
};
use image::{DynamicImage, imageops::FilterType};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::{
    fs,
    path::Path,
};

// Structure pour images
#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub data: Vec<f32>,
    pub label: usize,
    pub height: usize,
    pub width: usize,
}

impl PreprocessedImage {
    pub fn new(img: &DynamicImage, label: usize, size: (usize, usize)) -> Self {
        let (target_width, target_height) = (size.0 as u32, size.1 as u32);
        
        let img = img.resize_exact(target_width, target_height, FilterType::Lanczos3);
        let rgb_img = img.to_rgb8();
        
        let mut data = Vec::with_capacity(3 * target_height as usize * target_width as usize);
        
        for y in 0..target_height {
            for x in 0..target_width {
                let pixel = rgb_img.get_pixel(x, y);
                data.push(pixel[0] as f32 / 255.0);
                data.push(pixel[1] as f32 / 255.0);
                data.push(pixel[2] as f32 / 255.0);
            }
        }
        
        Self { 
            data, 
            label,
            height: target_height as usize,
            width: target_width as usize,
        }
    }
}

// Dataset
#[derive(Debug, Clone)]
pub struct MalariaDataset {
    pub items: Vec<PreprocessedImage>,
    pub image_size: (usize, usize),
}

impl MalariaDataset {
    pub fn new(data_path: &Path, image_size: (usize, usize)) -> anyhow::Result<Self> {
        let mut items = Vec::new();
        
        // Parasitized (label 0)
        let parasitized_path = data_path.join("Parasitized");
        if parasitized_path.exists() && parasitized_path.is_dir() {
            for entry in fs::read_dir(parasitized_path)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().map_or(false, |ext| ext == "png" || ext == "jpg") {
                    match image::open(&path) {
                        Ok(img) => {
                            items.push(PreprocessedImage::new(&img, 0, image_size));
                        }
                        Err(e) => eprintln!("⚠️ Erreur image {:?}: {}", path, e),
                    }
                }
            }
        }
        
        // Uninfected (label 1)
        let uninfected_path = data_path.join("Uninfected");
        if uninfected_path.exists() && uninfected_path.is_dir() {
            for entry in fs::read_dir(uninfected_path)? {
                let entry = entry?;
                let path = entry.path();
                
                if path.extension().map_or(false, |ext| ext == "png" || ext == "jpg") {
                    match image::open(&path) {
                        Ok(img) => {
                            items.push(PreprocessedImage::new(&img, 1, image_size));
                        }
                        Err(e) => eprintln!("⚠️ Erreur image {:?}: {}", path, e),
                    }
                }
            }
        }
        
        // Mélange aléatoire
        let mut rng = StdRng::seed_from_u64(42);
        items.shuffle(&mut rng);
        
        println!("✅ Dataset chargé: {} images", items.len());
        
        Ok(Self {
            items,
            image_size,
        })
    }
    
    pub fn len(&self) -> usize {
        self.items.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
    
    pub fn split(self, train_ratio: f64, val_ratio: f64) -> (Self, Self, Self) {
        let total = self.items.len();
        let train_end = (total as f64 * train_ratio) as usize;
        let val_end = train_end + (total as f64 * val_ratio) as usize;
        
        let items = self.items;
        
        (
            Self {
                items: items[0..train_end].to_vec(),
                image_size: self.image_size,
            },
            Self {
                items: items[train_end..val_end].to_vec(),
                image_size: self.image_size,
            },
            Self {
                items: items[val_end..].to_vec(),
                image_size: self.image_size,
            },
        )
    }
}

// Implémentation Dataset de Burn
impl Dataset<PreprocessedImage> for MalariaDataset {
    fn get(&self, index: usize) -> Option<PreprocessedImage> {
        self.items.get(index).cloned()
    }
    
    fn len(&self) -> usize {
        self.items.len()
    }
}

// Batch structure
#[derive(Debug, Clone)]
pub struct MalariaBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

// Batcher
#[derive(Debug)]
pub struct MalariaBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MalariaBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<B, PreprocessedImage, MalariaBatch<B>> for MalariaBatcher<B> {
    fn batch(&self, items: Vec<PreprocessedImage>, _device: &B::Device) -> MalariaBatch<B> {
        let batch_size = items.len();
        
        if batch_size == 0 {
            return MalariaBatch {
                images: Tensor::zeros([0, 3, 128, 128], &self.device),
                targets: Tensor::zeros([0], &self.device),
            };
        }
        
        let height = items[0].height;
        let width = items[0].width;
        
        let mut images_data = Vec::with_capacity(batch_size * 3 * height * width);
        let mut labels_data = Vec::with_capacity(batch_size);
        
        for item in items {
            images_data.extend_from_slice(&item.data);
            labels_data.push(item.label as i64);
        }
        
        let images_tensor = Tensor::from_data(
            TensorData::new(images_data, Shape::new([batch_size, 3, height, width])),
            &self.device,
        );
        
        let targets_tensor: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(labels_data, Shape::new([batch_size])),
            &self.device,
        );
        
        MalariaBatch { 
            images: images_tensor, 
            targets: targets_tensor 
        }
    }
}