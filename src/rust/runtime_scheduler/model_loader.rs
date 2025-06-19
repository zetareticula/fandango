// src/rust/runtime_scheduler/model_loader.rs

use crate::quantized_model_loader; // Module for loading GGUF/safetensors
use std::path::Path;

pub struct ModelLoader {
    loader: quantized_model_loader::QuantizedLoader, // Integration with quantized_model_loader
    cache_dir: String, // Directory for cached quantized layers
}

impl ModelLoader {
    pub fn new() -> Self {
        ModelLoader {
            loader: quantized_model_loader::QuantizedLoader::new(),
            cache_dir: String::from("cached_layers/"), // Default cache directory
        }
    }

    pub fn load_quantized(&self, model_id: &str) -> Vec<u8> {
        // Check cache first
        let cache_path = Path::new(&self.cache_dir).join(format!("{}.quantized", model_id));
        if cache_path.exists() {
            return std::fs::read(cache_path).unwrap_or_else(|e| {
                eprintln!("Cache read error for {}: {}", model_id, e);
                vec![]
            });
        }

        // Load from GGUF/safetensors if not in cache
        let file_path = Path::new("models/").join(format!("{}.gguf", model_id));
        match self.loader.load_from_file(&file_path) {
            Ok(raw_data) => {
                // Apply dynamic quantization (placeholder logic)
                let quantized_data: Vec<u8> = raw_data.into_iter().map(|f| {
                    let quantized = if f.abs() < 0.5 { (f * 15.0) as u8 } else { (f * 255.0) as u8 }; // 4-bit or 8-bit based on value
                    quantized
                }).collect();

                // Cache the quantized data
                if let Err(e) = std::fs::write(&cache_path, &quantized_data) {
                    eprintln!("Cache write error for {}: {}", model_id, e);
                }

                quantized_data
            }
            Err(e) => {
                eprintln!("Failed to load model {}: {}", model_id, e);
                vec![]
            }
        }
    }

    pub fn unload(&self, model_id: &str) {
        let cache_path = Path::new(&self.cache_dir).join(format!("{}.quantized", model_id));
        if cache_path.exists() {
            if let Err(e) = std::fs::remove_file(&cache_path) {
                eprintln!("Unload error for {}: {}", model_id, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_quantized() {
        let loader = ModelLoader::new();
        let data = loader.load_quantized("test_model");
        assert!(!data.is_empty(), "Loaded data should not be empty");
    }

    #[test]
    fn test_unload() {
        let loader = ModelLoader::new();
        loader.unload("test_model"); // Test cleanup
    }
}