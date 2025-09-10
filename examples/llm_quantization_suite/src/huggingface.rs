//! HuggingFace Integration - Safetensors and JSON model loading

use crate::{ModelConfig, QuantizationResult, QuantizationError};
use anyhow::Result;
use hf_hub::api::tokio::Api;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, debug, error};

/// HuggingFace model loader with safetensors support
pub struct HuggingFaceLoader {
    api: Api,
    cache_dir: PathBuf,
}

impl HuggingFaceLoader {
    pub fn new() -> Self {
        let api = Api::new().expect("Failed to initialize HuggingFace API");
        let cache_dir = std::env::temp_dir().join("fandango_hf_cache");
        
        Self { api, cache_dir }
    }

    /// Load complete model from HuggingFace repository
    pub async fn load_model(&self, config: &ModelConfig) -> QuantizationResult<ModelData> {
        info!("Loading model: {} (revision: {:?})", config.model_id, config.revision);

        // Create repository reference
        let repo = self.api.model(config.model_id.clone());
        let repo = if let Some(revision) = &config.revision {
            repo.revision(revision.clone())
        } else {
            repo
        };

        // Load model configuration
        let model_config = self.load_model_config(&repo, &config.config_json).await?;
        
        // Load safetensors files
        let mut tensors = HashMap::new();
        for safetensor_file in &config.safetensors_files {
            let tensor_data = self.load_safetensors(&repo, safetensor_file).await?;
            tensors.extend(tensor_data);
        }

        // Load tokenizer if specified
        let tokenizer_config = if let Some(tokenizer_path) = &config.tokenizer_config {
            Some(self.load_tokenizer_config(&repo, tokenizer_path).await?)
        } else {
            None
        };

        // Validate loaded model
        self.validate_model_structure(&tensors, config).await?;

        Ok(ModelData {
            config: model_config,
            tensors,
            tokenizer_config,
            model_id: config.model_id.clone(),
            revision: config.revision.clone(),
        })
    }

    /// Load model configuration JSON
    async fn load_model_config(&self, repo: &hf_hub::api::tokio::RepoInfo, config_path: &str) -> QuantizationResult<Value> {
        debug!("Loading model config: {}", config_path);
        
        let config_file = repo.get(config_path).await
            .map_err(|e| QuantizationError::HuggingFaceError(format!("Failed to download config: {}", e)))?;
        
        let config_content = fs::read_to_string(&config_file).await
            .map_err(|e| QuantizationError::ModelLoadError(format!("Failed to read config: {}", e)))?;
        
        serde_json::from_str(&config_content)
            .map_err(|e| QuantizationError::ModelLoadError(format!("Invalid config JSON: {}", e)))
    }

    /// Load safetensors file and extract tensors
    async fn load_safetensors(&self, repo: &hf_hub::api::tokio::RepoInfo, safetensor_path: &str) -> QuantizationResult<HashMap<String, TensorData>> {
        debug!("Loading safetensors: {}", safetensor_path);
        
        let safetensor_file = repo.get(safetensor_path).await
            .map_err(|e| QuantizationError::HuggingFaceError(format!("Failed to download safetensors: {}", e)))?;
        
        let safetensor_bytes = fs::read(&safetensor_file).await
            .map_err(|e| QuantizationError::ModelLoadError(format!("Failed to read safetensors: {}", e)))?;
        
        let safetensors = SafeTensors::deserialize(&safetensor_bytes)
            .map_err(|e| QuantizationError::ModelLoadError(format!("Invalid safetensors format: {}", e)))?;
        
        let mut tensors = HashMap::new();
        
        for tensor_name in safetensors.names() {
            let tensor_view = safetensors.tensor(tensor_name)
                .map_err(|e| QuantizationError::ModelLoadError(format!("Failed to get tensor {}: {}", tensor_name, e)))?;
            
            let tensor_data = TensorData {
                name: tensor_name.to_string(),
                shape: tensor_view.shape().to_vec(),
                dtype: tensor_view.dtype(),
                data: tensor_view.data().to_vec(),
            };
            
            tensors.insert(tensor_name.to_string(), tensor_data);
        }
        
        info!("Loaded {} tensors from {}", tensors.len(), safetensor_path);
        Ok(tensors)
    }

    /// Load tokenizer configuration
    async fn load_tokenizer_config(&self, repo: &hf_hub::api::tokio::RepoInfo, tokenizer_path: &str) -> QuantizationResult<Value> {
        debug!("Loading tokenizer config: {}", tokenizer_path);
        
        let tokenizer_file = repo.get(tokenizer_path).await
            .map_err(|e| QuantizationError::HuggingFaceError(format!("Failed to download tokenizer: {}", e)))?;
        
        let tokenizer_content = fs::read_to_string(&tokenizer_file).await
            .map_err(|e| QuantizationError::ModelLoadError(format!("Failed to read tokenizer: {}", e)))?;
        
        serde_json::from_str(&tokenizer_content)
            .map_err(|e| QuantizationError::ModelLoadError(format!("Invalid tokenizer JSON: {}", e)))
    }

    /// Validate model structure against expected configuration
    async fn validate_model_structure(&self, tensors: &HashMap<String, TensorData>, config: &ModelConfig) -> QuantizationResult<()> {
        debug!("Validating model structure");

        // Count parameters
        let total_params: u64 = tensors.values()
            .map(|tensor| tensor.shape.iter().product::<usize>() as u64)
            .sum();

        if total_params != config.expected_params {
            return Err(QuantizationError::ValidationFailed(
                format!("Parameter count mismatch: expected {}, got {}", config.expected_params, total_params)
            ));
        }

        // Count layers (approximate by counting weight matrices)
        let layer_count = tensors.keys()
            .filter(|name| name.contains("weight") && (name.contains("mlp") || name.contains("attn")))
            .count();

        if layer_count < config.expected_layers {
            return Err(QuantizationError::ValidationFailed(
                format!("Layer count too low: expected at least {}, found {}", config.expected_layers, layer_count)
            ));
        }

        info!("Model validation passed: {} parameters, {} layers", total_params, layer_count);
        Ok(())
    }
}

/// Loaded model data structure
#[derive(Debug)]
pub struct ModelData {
    pub config: Value,
    pub tensors: HashMap<String, TensorData>,
    pub tokenizer_config: Option<Value>,
    pub model_id: String,
    pub revision: Option<String>,
}

impl ModelData {
    /// Calculate total model size in MB
    pub fn calculate_size_mb(&self) -> f64 {
        let total_bytes: usize = self.tensors.values()
            .map(|tensor| tensor.data.len())
            .sum();
        total_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get MLP layer tensors for concurrent quantization
    pub fn get_mlp_layers(&self) -> Vec<(&String, &TensorData)> {
        self.tensors.iter()
            .filter(|(name, _)| {
                name.contains("mlp") && (
                    name.contains("up_proj") || 
                    name.contains("down_proj") || 
                    name.contains("gate_proj") ||
                    name.contains("fc1") ||
                    name.contains("fc2")
                )
            })
            .collect()
    }

    /// Get attention layer tensors
    pub fn get_attention_layers(&self) -> Vec<(&String, &TensorData)> {
        self.tensors.iter()
            .filter(|(name, _)| {
                name.contains("attn") && (
                    name.contains("q_proj") || 
                    name.contains("k_proj") || 
                    name.contains("v_proj") ||
                    name.contains("o_proj")
                )
            })
            .collect()
    }

    /// Extract model architecture information
    pub fn get_architecture_info(&self) -> ModelArchitecture {
        let config = &self.config;
        
        ModelArchitecture {
            model_type: config.get("model_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            hidden_size: config.get("hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            num_layers: config.get("num_hidden_layers")
                .or_else(|| config.get("num_layers"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            num_attention_heads: config.get("num_attention_heads")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            intermediate_size: config.get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            vocab_size: config.get("vocab_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
        }
    }
}

/// Individual tensor data
#[derive(Debug, Clone)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
    pub data: Vec<u8>,
}

impl TensorData {
    /// Get tensor size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is a weight matrix (2D)
    pub fn is_weight_matrix(&self) -> bool {
        self.shape.len() == 2
    }

    /// Check if tensor belongs to MLP layer
    pub fn is_mlp_tensor(&self) -> bool {
        self.name.contains("mlp") || 
        self.name.contains("fc") || 
        self.name.contains("feed_forward")
    }
}

/// Model architecture information
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
}

/// Predefined model configurations for common HuggingFace models
pub struct ModelConfigs;

impl ModelConfigs {
    /// Get configuration for LLaMA-2 7B model
    pub fn llama2_7b() -> ModelConfig {
        ModelConfig {
            model_id: "meta-llama/Llama-2-7b-hf".to_string(),
            revision: None,
            safetensors_files: vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ],
            config_json: "config.json".to_string(),
            tokenizer_config: Some("tokenizer_config.json".to_string()),
            expected_layers: 32,
            expected_params: 6_738_415_616,
        }
    }

    /// Get configuration for Mistral 7B model
    pub fn mistral_7b() -> ModelConfig {
        ModelConfig {
            model_id: "mistralai/Mistral-7B-v0.1".to_string(),
            revision: None,
            safetensors_files: vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ],
            config_json: "config.json".to_string(),
            tokenizer_config: Some("tokenizer_config.json".to_string()),
            expected_layers: 32,
            expected_params: 7_241_732_096,
        }
    }

    /// Get configuration for CodeLlama 7B model
    pub fn codellama_7b() -> ModelConfig {
        ModelConfig {
            model_id: "codellama/CodeLlama-7b-hf".to_string(),
            revision: None,
            safetensors_files: vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ],
            config_json: "config.json".to_string(),
            tokenizer_config: Some("tokenizer_config.json".to_string()),
            expected_layers: 32,
            expected_params: 6_738_415_616,
        }
    }
}
