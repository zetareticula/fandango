
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

//! Quantization block implementation

use serde::{Serialize, Deserialize};
use ndarray::{ArrayD, ArrayViewD, Array};
use num_traits::Float;
use super::*;

// Helper function to calculate the scale and zero point for quantization
fn calculate_scale_zero_point<T: Float>(
    min: T,
    max: T,
    qmin: i32,
    qmax: i32,
) -> (T, T) {
    let scale = (max - min) / T::from(qmax - qmin).unwrap();
    let zero_point = T::from(qmin).unwrap() - (min / scale).round();
    (scale, zero_point)
}

// Helper function to quantize a single value
fn quantize_value<T: Float>(
    value: T,
    scale: T,
    zero_point: T,
    qmin: i32,
    qmax: i32,
) -> i32 {
    let q = (value / scale + zero_point).round();
    q.max(T::from(qmin).unwrap())
        .min(T::from(qmax).unwrap())
        .to_i32()
        .unwrap()
}

// Helper function to dequantize a single value
fn dequantize_value<T: Float>(q: i32, scale: T, zero_point: T) -> T {
    scale * (T::from(q).unwrap() - zero_point)
}

/// A block that applies quantization to model weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationBlock {
    bits: u8,
    group_size: usize,
    symmetric: bool,
}

impl QuantizationBlock {
    /// Create a new quantization block
    pub fn new(bits: u8) -> Self {
        Self {
            bits,
            group_size: 128,  // Default group size
            symmetric: true,  // Default to symmetric quantization
        }
    }
    
    /// Set the group size
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;
        self
    }
    
    /// Set whether to use symmetric quantization
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
}

impl OptimizationBlock for QuantizationBlock {
    fn block_type(&self) -> &'static str {
        "quantization"
    }
    
    fn name(&self) -> &'static str {
        "Quantization"
    }
    
    fn description(&self) -> &'static str {
        "Applies quantization to model weights to reduce memory usage and improve inference speed"
    }
    
    fn category(&self) -> &'static str {
        "Quantization"
    }
    
    fn inputs(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("weights", "tensor[f32]"),
            ("scale", "tensor[f32] (optional)"),
            ("zero_point", "tensor[f32] (optional)"),
        ]
    }
    
    fn outputs(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("quantized_weights", "tensor[int]"),
            ("scale", "tensor[f32]"),
            ("zero_point", "tensor[f32] (if not symmetric)"),
        ]
    }
    
    fn process<'a>(
        &'a mut self,
        inputs: BlockInputs,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BlockOutputs>> + Send + 'a>> {
        // Clone the fields we need for the async block
        let bits = self.bits;
        let group_size = self.group_size;
        let symmetric = self.symmetric;
        
        Box::pin(async move {
            // Get input tensors
            let weights_json = inputs.values.get("weights")
                .ok_or_else(|| WorkspaceError::BlockError("Missing 'weights' input".to_string()))?;
            
            // Convert JSON to ndarray
            let weights: Vec<f32> = serde_json::from_value(weights_json.clone())
                .map_err(|e| WorkspaceError::BlockError(format!("Invalid weights format: {}", e)))?;
            
            let len = weights.len();
            let num_groups = (len + group_size - 1) / group_size;
            
            // Calculate quantization parameters
            let qmin = 0;
            let qmax = (1 << bits) - 1;
            
            let mut quantized_weights = Vec::with_capacity(len);
            let mut scales = Vec::with_capacity(num_groups);
            let mut zero_points = if symmetric { None } else { Some(Vec::with_capacity(num_groups)) };
            
            // Process each group
            for group_idx in 0..num_groups {
                let start = group_idx * group_size;
                let end = (start + group_size).min(len);
                let group = &weights[start..end];
                
                // Find min and max in the group
                let min = *group.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
                let max = *group.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
                
                // Calculate scale and zero point
                let (scale, zero_point) = if symmetric {
                    let max_val = max.abs().max(min.abs());
                    (max_val / (qmax as f32), 0.0)
                } else {
                    calculate_scale_zero_point(min, max, qmin, qmax)
                };
                
                // Quantize the group
                for &value in group {
                    let q = quantize_value(value, scale, zero_point, qmin, qmax);
                    quantized_weights.push(q);
                }
                
                scales.push(scale);
                if let Some(ref mut zps) = zero_points {
                    zps.push(zero_point);
                }
            }
            
            // Prepare outputs
            let mut outputs = BlockOutputs::default();
            outputs.values.insert("quantized_weights".to_string(), serde_json::json!(quantized_weights));
            outputs.values.insert("scale".to_string(), serde_json::json!(scales));
            
            // Handle zero points and metrics
            if !symmetric {
                let zero_points = zero_points.unwrap();
                let compressed_size = (len * bits as usize / 8) + 
                    (scales.len() * std::mem::size_of::<f32>()) +
                    (zero_points.len() * std::mem::size_of::<f32>());
                let compression_ratio = (len * std::mem::size_of::<f32>()) as f64 / compressed_size as f64;
                
                outputs.values.insert("zero_point".to_string(), serde_json::json!(zero_points));
                
                // Add quantization metrics
                outputs.values.insert("quantization_metrics".to_string(), serde_json::json!({
                    "bits": bits,
                    "original_size_bytes": len * std::mem::size_of::<f32>(),
                    "compressed_size_bytes": compressed_size,
                    "compression_ratio": compression_ratio,
                }));
            } else {
                let compressed_size = (len * bits as usize / 8) + 
                    (scales.len() * std::mem::size_of::<f32>());
                let compression_ratio = (len * std::mem::size_of::<f32>()) as f64 / compressed_size as f64;
                
                // Add quantization metrics for symmetric case
                outputs.values.insert("quantization_metrics".to_string(), serde_json::json!({
                    "bits": bits,
                    "original_size_bytes": len * std::mem::size_of::<f32>(),
                    "compressed_size_bytes": compressed_size,
                    "compression_ratio": compression_ratio,
                }));
            }
            
            Ok(outputs)
        })
    }
    
    fn clone_box(&self) -> Box<dyn OptimizationBlock> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_quantization_block_symmetric() {
        // Create a quantization block with 4-bit symmetric quantization
        let mut block = QuantizationBlock::new(4).with_group_size(4);
        
        // Create test input weights
        let mut inputs = BlockInputs::default();
        let test_weights = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        inputs.values.insert("weights".to_string(), json!(test_weights));
        
        // Process the inputs
        let result = block.process(inputs).await.unwrap();
        
        // Verify the outputs
        assert!(result.values.contains_key("quantized_weights"));
        assert!(result.values.contains_key("scale"));
        assert!(!result.values.contains_key("zero_point"), "Should not have zero_point in symmetric mode");
        
        // Verify metrics
        assert!(result.values.contains_key("quantization_metrics"));
        let metrics = result.values.get("quantization_metrics").unwrap();
        assert_eq!(metrics["bits"], 4);
        assert!(metrics["compression_ratio"].as_f64().unwrap() > 1.0, "Compression ratio should be greater than 1.0");
    }

    #[tokio::test]
    async fn test_quantization_block_asymmetric() {
        // Create a quantization block with 4-bit asymmetric quantization
        let mut block = QuantizationBlock::new(4)
            .with_group_size(4)
            .with_symmetric(false);
        
        // Create test input weights
        let mut inputs = BlockInputs::default();
        let test_weights = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        inputs.values.insert("weights".to_string(), json!(test_weights));
        
        // Process the inputs
        let result = block.process(inputs).await.unwrap();
        
        // Verify the outputs
        assert!(result.values.contains_key("quantized_weights"));
        assert!(result.values.contains_key("scale"));
        assert!(result.values.contains_key("zero_point"), "Should have zero_point in asymmetric mode");
        
        // Verify metrics
        assert!(result.values.contains_key("quantization_metrics"));
        let metrics = result.values.get("quantization_metrics").unwrap();
        assert_eq!(metrics["bits"], 4);
        assert!(metrics["compression_ratio"].as_f64().unwrap() > 1.0, "Compression ratio should be greater than 1.0");
    }

    #[tokio::test]
    async fn test_quantization_block() {
        let mut block = QuantizationBlock::new(4);
        let mut inputs = BlockInputs::default();
        inputs.values.insert("weights".to_string(), serde_json::json!([1.0, 2.0, 3.0, 4.0]));
        
        let outputs = block.process(inputs).await.unwrap();
        
        assert!(outputs.values.contains_key("quantized_weights"));
        assert!(outputs.values.contains_key("scale"));
    }
}
