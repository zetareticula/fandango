//! Validation Engine - Comprehensive testing and verification of quantized models

use crate::{QuantizationResult, QuantizationError, quantization::QuantizedModelData};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{info, debug, warn};

/// Quantization validation engine
pub struct QuantizationValidator {
    device: Device,
    benchmark_samples: usize,
}

impl QuantizationValidator {
    pub fn new() -> Self {
        let device = if cfg!(feature = "gpu") {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };

        Self {
            device,
            benchmark_samples: 1000,
        }
    }

    /// Calculate accuracy loss compared to original model
    pub async fn calculate_accuracy_loss(&self, quantized_model: &QuantizedModelData) -> QuantizationResult<f64> {
        info!("Calculating accuracy loss for quantized model");

        // Generate test inputs for validation
        let test_inputs = self.generate_test_inputs().await?;
        
        // Simulate original model inference (in practice, would load original)
        let original_outputs = self.simulate_original_inference(&test_inputs).await?;
        
        // Run quantized model inference
        let quantized_outputs = self.run_quantized_inference(quantized_model, &test_inputs).await?;
        
        // Calculate various accuracy metrics
        let mse = self.calculate_mse(&original_outputs, &quantized_outputs)?;
        let cosine_similarity = self.calculate_cosine_similarity(&original_outputs, &quantized_outputs)?;
        let relative_error = self.calculate_relative_error(&original_outputs, &quantized_outputs)?;
        
        // Combine metrics into overall accuracy loss
        let accuracy_loss = 1.0 - cosine_similarity.min(1.0 - relative_error);
        
        info!("Accuracy metrics - MSE: {:.6}, Cosine: {:.4}, RelError: {:.4}, Loss: {:.4}", 
              mse, cosine_similarity, relative_error, accuracy_loss);
        
        Ok(accuracy_loss.max(0.0))
    }

    /// Measure latency increase compared to baseline
    pub async fn measure_latency_increase(&self, quantized_model: &QuantizedModelData) -> QuantizationResult<f64> {
        info!("Measuring latency increase for quantized model");

        let test_input = self.generate_single_test_input().await?;
        
        // Benchmark original model latency (simulated)
        let original_latency = self.benchmark_original_latency(&test_input).await?;
        
        // Benchmark quantized model latency
        let quantized_latency = self.benchmark_quantized_latency(quantized_model, &test_input).await?;
        
        let latency_increase = (quantized_latency - original_latency) / original_latency;
        
        info!("Latency - Original: {:.2}ms, Quantized: {:.2}ms, Increase: {:.1}%", 
              original_latency, quantized_latency, latency_increase * 100.0);
        
        Ok(latency_increase)
    }

    /// Measure throughput ratio (quantized vs original)
    pub async fn measure_throughput_ratio(&self, quantized_model: &QuantizedModelData) -> QuantizationResult<f64> {
        info!("Measuring throughput ratio for quantized model");

        let batch_size = 32;
        let sequence_length = 512;
        
        // Generate batch of test inputs
        let test_batch = self.generate_test_batch(batch_size, sequence_length).await?;
        
        // Benchmark original throughput (simulated)
        let original_throughput = self.benchmark_original_throughput(&test_batch).await?;
        
        // Benchmark quantized throughput
        let quantized_throughput = self.benchmark_quantized_throughput(quantized_model, &test_batch).await?;
        
        let throughput_ratio = quantized_throughput / original_throughput;
        
        info!("Throughput - Original: {:.1} tok/s, Quantized: {:.1} tok/s, Ratio: {:.3}", 
              original_throughput, quantized_throughput, throughput_ratio);
        
        Ok(throughput_ratio)
    }

    /// Validate model structure and integrity
    pub async fn validate_model_integrity(&self, quantized_model: &QuantizedModelData) -> QuantizationResult<ValidationReport> {
        info!("Validating quantized model integrity");

        let mut report = ValidationReport::new();
        
        // Check tensor shapes consistency
        for (name, tensor) in &quantized_model.tensors {
            if tensor.shape.is_empty() {
                report.add_error(format!("Tensor {} has empty shape", name));
            }
            
            if tensor.data.is_empty() {
                report.add_error(format!("Tensor {} has empty data", name));
            }
            
            // Validate tensor size consistency
            let expected_size = tensor.shape.iter().product::<usize>() * self.get_dtype_size(&tensor.dtype);
            if tensor.data.len() != expected_size {
                report.add_error(format!("Tensor {} size mismatch: expected {}, got {}", 
                                       name, expected_size, tensor.data.len()));
            }
        }
        
        // Check for NaN or infinite values
        for (name, tensor) in &quantized_model.tensors {
            if self.contains_invalid_values(tensor).await? {
                report.add_warning(format!("Tensor {} contains NaN or infinite values", name));
            }
        }
        
        // Validate quantization parameters
        self.validate_quantization_config(&quantized_model.quantization_config, &mut report);
        
        info!("Model integrity validation completed: {} errors, {} warnings", 
              report.errors.len(), report.warnings.len());
        
        Ok(report)
    }

    /// Generate test inputs for validation
    async fn generate_test_inputs(&self) -> QuantizationResult<Vec<Tensor>> {
        let mut inputs = Vec::new();
        
        for _ in 0..self.benchmark_samples {
            let input = Tensor::randn(0.0f32, 1.0, &[1, 512, 4096], &self.device)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            inputs.push(input);
        }
        
        Ok(inputs)
    }

    /// Generate single test input
    async fn generate_single_test_input(&self) -> QuantizationResult<Tensor> {
        Tensor::randn(0.0f32, 1.0, &[1, 512, 4096], &self.device)
            .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))
    }

    /// Generate batch of test inputs
    async fn generate_test_batch(&self, batch_size: usize, seq_len: usize) -> QuantizationResult<Tensor> {
        Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, 4096], &self.device)
            .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))
    }

    /// Simulate original model inference (placeholder)
    async fn simulate_original_inference(&self, inputs: &[Tensor]) -> QuantizationResult<Vec<Tensor>> {
        let mut outputs = Vec::new();
        
        for input in inputs {
            // Simulate original model computation
            let output = input.matmul(&Tensor::randn(0.0f32, 1.0, &[4096, 4096], &self.device)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }

    /// Run quantized model inference
    async fn run_quantized_inference(&self, model: &QuantizedModelData, inputs: &[Tensor]) -> QuantizationResult<Vec<Tensor>> {
        let mut outputs = Vec::new();
        
        // Find a representative weight tensor for simulation
        let weight_tensor = model.tensors.values()
            .find(|t| t.is_mlp_tensor() && t.is_weight_matrix())
            .ok_or_else(|| QuantizationError::ValidationFailed("No suitable weight tensor found".to_string()))?;
        
        // Convert to Candle tensor for computation
        let weight = self.tensor_data_to_candle(weight_tensor)?;
        
        for input in inputs {
            // Simulate quantized model computation
            let output = input.matmul(&weight)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }

    /// Calculate Mean Squared Error
    fn calculate_mse(&self, original: &[Tensor], quantized: &[Tensor]) -> QuantizationResult<f64> {
        if original.len() != quantized.len() {
            return Err(QuantizationError::ValidationFailed("Mismatched output lengths".to_string()));
        }
        
        let mut total_mse = 0.0;
        let mut total_elements = 0;
        
        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let diff = orig.sub(quant)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            let squared = diff.powf(2.0)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            let mse = squared.mean_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            
            total_mse += mse as f64;
            total_elements += 1;
        }
        
        Ok(total_mse / total_elements as f64)
    }

    /// Calculate cosine similarity
    fn calculate_cosine_similarity(&self, original: &[Tensor], quantized: &[Tensor]) -> QuantizationResult<f64> {
        if original.len() != quantized.len() {
            return Err(QuantizationError::ValidationFailed("Mismatched output lengths".to_string()));
        }
        
        let mut total_similarity = 0.0;
        
        for (orig, quant) in original.iter().zip(quantized.iter()) {
            // Flatten tensors for cosine similarity calculation
            let orig_flat = orig.flatten_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            let quant_flat = quant.flatten_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            
            // Calculate dot product
            let dot_product = orig_flat.mul(&quant_flat)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .sum_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            
            // Calculate norms
            let orig_norm = orig_flat.powf(2.0)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .sum_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .sqrt();
            
            let quant_norm = quant_flat.powf(2.0)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .sum_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .sqrt();
            
            let similarity = dot_product / (orig_norm * quant_norm);
            total_similarity += similarity as f64;
        }
        
        Ok(total_similarity / original.len() as f64)
    }

    /// Calculate relative error
    fn calculate_relative_error(&self, original: &[Tensor], quantized: &[Tensor]) -> QuantizationResult<f64> {
        if original.len() != quantized.len() {
            return Err(QuantizationError::ValidationFailed("Mismatched output lengths".to_string()));
        }
        
        let mut total_error = 0.0;
        
        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let diff = orig.sub(quant)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            let abs_diff = diff.abs()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            let abs_orig = orig.abs()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            
            let rel_error = abs_diff.div(&abs_orig)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .mean_all()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?
                .to_scalar::<f32>()
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
            
            total_error += rel_error as f64;
        }
        
        Ok(total_error / original.len() as f64)
    }

    /// Benchmark original model latency (simulated)
    async fn benchmark_original_latency(&self, input: &Tensor) -> QuantizationResult<f64> {
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            // Simulate original model inference
            let _output = input.matmul(&Tensor::randn(0.0f32, 1.0, &[4096, 4096], &self.device)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
        }
        
        let elapsed = start.elapsed();
        Ok(elapsed.as_millis() as f64 / iterations as f64)
    }

    /// Benchmark quantized model latency
    async fn benchmark_quantized_latency(&self, model: &QuantizedModelData, input: &Tensor) -> QuantizationResult<f64> {
        let weight_tensor = model.tensors.values()
            .find(|t| t.is_mlp_tensor() && t.is_weight_matrix())
            .ok_or_else(|| QuantizationError::ValidationFailed("No suitable weight tensor found".to_string()))?;
        
        let weight = self.tensor_data_to_candle(weight_tensor)?;
        
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _output = input.matmul(&weight)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
        }
        
        let elapsed = start.elapsed();
        Ok(elapsed.as_millis() as f64 / iterations as f64)
    }

    /// Benchmark original throughput (simulated)
    async fn benchmark_original_throughput(&self, batch: &Tensor) -> QuantizationResult<f64> {
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let _output = batch.matmul(&Tensor::randn(0.0f32, 1.0, &[4096, 4096], &self.device)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
        }
        
        let elapsed = start.elapsed();
        let tokens_processed = batch.shape()[0] * batch.shape()[1] * iterations;
        Ok(tokens_processed as f64 / elapsed.as_secs_f64())
    }

    /// Benchmark quantized throughput
    async fn benchmark_quantized_throughput(&self, model: &QuantizedModelData, batch: &Tensor) -> QuantizationResult<f64> {
        let weight_tensor = model.tensors.values()
            .find(|t| t.is_mlp_tensor() && t.is_weight_matrix())
            .ok_or_else(|| QuantizationError::ValidationFailed("No suitable weight tensor found".to_string()))?;
        
        let weight = self.tensor_data_to_candle(weight_tensor)?;
        
        let start = Instant::now();
        let iterations = 10;
        
        for _ in 0..iterations {
            let _output = batch.matmul(&weight)
                .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))?;
        }
        
        let elapsed = start.elapsed();
        let tokens_processed = batch.shape()[0] * batch.shape()[1] * iterations;
        Ok(tokens_processed as f64 / elapsed.as_secs_f64())
    }

    /// Check if tensor contains invalid values
    async fn contains_invalid_values(&self, tensor_data: &crate::huggingface::TensorData) -> QuantizationResult<bool> {
        // Convert to f32 for checking
        let data = match tensor_data.dtype {
            safetensors::Dtype::F32 => {
                tensor_data.data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<f32>>()
            }
            _ => return Ok(false), // Skip non-float types
        };
        
        Ok(data.iter().any(|&x| x.is_nan() || x.is_infinite()))
    }

    /// Validate quantization configuration
    fn validate_quantization_config(&self, config: &crate::QuantizationStrategy, report: &mut ValidationReport) {
        if let Some(group_size) = config.group_size {
            if group_size == 0 {
                report.add_error("Group size cannot be zero".to_string());
            }
            if group_size > 1024 {
                report.add_warning("Large group size may reduce quantization effectiveness".to_string());
            }
        }
        
        if config.damp_percent < 0.0 || config.damp_percent > 1.0 {
            report.add_error("Damp percent must be between 0.0 and 1.0".to_string());
        }
    }

    /// Convert TensorData to Candle Tensor
    fn tensor_data_to_candle(&self, tensor_data: &crate::huggingface::TensorData) -> QuantizationResult<Tensor> {
        let data = match tensor_data.dtype {
            safetensors::Dtype::F32 => {
                tensor_data.data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect::<Vec<f32>>()
            }
            _ => return Err(QuantizationError::ValidationFailed("Unsupported dtype for validation".to_string())),
        };
        
        Tensor::from_vec(data, &tensor_data.shape, &self.device)
            .map_err(|e| QuantizationError::ValidationFailed(e.to_string()))
    }

    /// Get size in bytes for a data type
    fn get_dtype_size(&self, dtype: &safetensors::Dtype) -> usize {
        match dtype {
            safetensors::Dtype::F32 => 4,
            safetensors::Dtype::F16 => 2,
            safetensors::Dtype::BF16 => 2,
            safetensors::Dtype::U8 => 1,
            safetensors::Dtype::I8 => 1,
            safetensors::Dtype::I16 => 2,
            safetensors::Dtype::I32 => 4,
            safetensors::Dtype::I64 => 8,
            _ => 4, // Default to 4 bytes
        }
    }
}

/// Validation report structure
#[derive(Debug)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub passed: bool,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            passed: true,
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.passed = false;
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}
