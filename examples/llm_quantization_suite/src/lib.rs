//! LLM Quantization Suite - Container Circuit Proposition Framework
//! 
//! This module provides a comprehensive test suite for quantizing Large Language Models
//! using Fandango's serverless orchestrated concurrent MLP engineering capabilities.

pub mod circuit;
pub mod huggingface;
pub mod quantization;
pub mod orchestration;
pub mod validation;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Container Circuit Proposition - Core abstraction for testable quantization workflows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerCircuit {
    pub id: Uuid,
    pub name: String,
    pub model_config: ModelConfig,
    pub quantization_strategy: QuantizationStrategy,
    pub orchestration_config: OrchestrationConfig,
    pub validation_rules: Vec<ValidationRule>,
}

/// Model configuration for HuggingFace integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub revision: Option<String>,
    pub safetensors_files: Vec<String>,
    pub config_json: String,
    pub tokenizer_config: Option<String>,
    pub expected_layers: usize,
    pub expected_params: u64,
}

/// Quantization strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationStrategy {
    pub precision: QuantizationPrecision,
    pub group_size: Option<usize>,
    pub act_order: bool,
    pub damp_percent: f32,
    pub desc_act: bool,
    pub static_groups: bool,
    pub sym: bool,
    pub true_sequential: bool,
}

/// Quantization precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    Int2,
    Int4,
    Int8,
    Float16,
    BFloat16,
}

/// Orchestration configuration for serverless concurrent processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    pub max_concurrent_layers: usize,
    pub chunk_size: usize,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub memory_limit_gb: f32,
    pub cpu_cores: usize,
}

/// Validation rules for circuit propositions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub rule_type: ValidationType,
    pub threshold: f64,
    pub critical: bool,
}

/// Types of validation checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    CompressionRatio,
    AccuracyLoss,
    LatencyIncrease,
    MemoryReduction,
    ThroughputMaintenance,
}

/// Circuit execution result
#[derive(Debug, Serialize, Deserialize)]
pub struct CircuitResult {
    pub circuit_id: Uuid,
    pub success: bool,
    pub execution_time_ms: u64,
    pub metrics: CircuitMetrics,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Comprehensive metrics for circuit execution
#[derive(Debug, Serialize, Deserialize)]
pub struct CircuitMetrics {
    pub original_model_size_mb: f64,
    pub quantized_model_size_mb: f64,
    pub compression_ratio: f64,
    pub quantization_time_ms: u64,
    pub validation_time_ms: u64,
    pub memory_peak_mb: f64,
    pub throughput_tokens_per_sec: f64,
    pub accuracy_metrics: HashMap<String, f64>,
}

/// Main circuit executor trait
pub trait CircuitExecutor {
    async fn execute_circuit(&self, circuit: &ContainerCircuit) -> Result<CircuitResult>;
    async fn validate_proposition(&self, circuit: &ContainerCircuit) -> Result<bool>;
    async fn benchmark_performance(&self, circuit: &ContainerCircuit) -> Result<CircuitMetrics>;
}

/// Error types for the quantization suite
#[derive(thiserror::Error, Debug)]
pub enum QuantizationError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Quantization failed: {0}")]
    QuantizationFailed(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Orchestration error: {0}")]
    OrchestrationError(String),
    
    #[error("HuggingFace API error: {0}")]
    HuggingFaceError(String),
    
    #[error("Circuit proposition invalid: {0}")]
    InvalidCircuit(String),
}

/// Result type alias for convenience
pub type QuantizationResult<T> = Result<T, QuantizationError>;
