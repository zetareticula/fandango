use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear, Module, VarBuilder, VarMap};
use std::path::Path;
use thiserror::Error;
use anyhow::Context;

#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Shape mismatch in attention")]
    ShapeMismatch,
    #[error("Invalid input length")]
    InvalidInput,
    #[error(transparent)]
    CandleError(#[from] candle_core::Error),
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

type Result<T> = std::result::Result<T, AttentionError>;

pub struct FusedAttention {
    device: Device,
    input_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    dropout: f32,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl FusedAttention {
    pub fn new(
        device: Device,
        input_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        dropout: f32,
        weights_path: Option<&str>,
    ) -> Result<Self> {
        let mut var_map = VarMap::new();
        
        // Create a VarBuilder for the variables
        let vs = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Initialize weights using VarBuilder
        let q_weight = vs.get((hidden_dim, input_dim), "q_proj.weight")?;
        let q_bias = vs.get(hidden_dim, "q_proj.bias")?;
        
        let k_weight = vs.get((hidden_dim, input_dim), "k_proj.weight")?;
        let k_bias = vs.get(hidden_dim, "k_proj.bias")?;
        
        let v_weight = vs.get((hidden_dim, input_dim), "v_proj.weight")?;
        let v_bias = vs.get(hidden_dim, "v_proj.bias")?;
        
        let out_weight = vs.get((input_dim, hidden_dim), "out_proj.weight")?;
        let out_bias = vs.get(input_dim, "out_proj.bias")?;
        
        // Initialize the linear layers
        let _q_proj = Linear::new(q_weight, Some(q_bias));
        let _k_proj = Linear::new(k_weight, Some(k_bias));
        let _v_proj = Linear::new(v_weight, Some(v_bias));
        let _out_proj = Linear::new(out_weight, Some(out_bias));
        
        // Load pretrained weights if provided
        if let Some(path) = weights_path {
            let path = Path::new(path);
            var_map.load(path).context("Failed to load weights")?;
        }
        
        let _vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        
        // Initialize weights and biases with proper shape and device
        let q_weight = vs.get((hidden_dim, input_dim), "q_proj.weight")?;
        let q_bias = vs.get(hidden_dim, "q_proj.bias")?;
        let q_proj = Linear::new(q_weight, Some(q_bias));
        
        let k_weight = vs.get((hidden_dim, input_dim), "k_proj.weight")?;
        let k_bias = vs.get(hidden_dim, "k_proj.bias")?;
        let k_proj = Linear::new(k_weight, Some(k_bias));
        
        let v_weight = vs.get((hidden_dim, input_dim), "v_proj.weight")?;
        let v_bias = vs.get(hidden_dim, "v_proj.bias")?;
        let v_proj = Linear::new(v_weight, Some(v_bias));
        
        let out_weight = vs.get((input_dim, hidden_dim), "out_proj.weight")?;
        let out_bias = vs.get(input_dim, "out_proj.bias")?;
        let out_proj = Linear::new(out_weight, Some(out_bias));

        Ok(Self {
            device,
            input_dim,
            hidden_dim,
            num_heads,
            dropout,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
        })
    }

    pub async fn process_mermaid_flow(&mut self, input: Vec<f32>) -> Result<Vec<f32>> {
        // Validate input length
        if input.len() != self.input_dim {
            return Err(AttentionError::InvalidInput);
        }
        
        // Convert input to tensor
        let input_tensor = Tensor::from_vec(input, (1, self.input_dim), &self.device)?;
        
        // Project input to query, key, value
        let q = self.q_proj.forward(&input_tensor)?;
        let k = self.k_proj.forward(&input_tensor)?;
        let v = self.v_proj.forward(&input_tensor)?;
        
        // Compute attention scores
        let head_dim = self.hidden_dim / self.num_heads;
        let q = q.reshape((1, 1, self.num_heads, head_dim))?.transpose(1, 2)?; // [batch, num_heads, seq_len, head_dim]
        let k = k.reshape((1, 1, self.num_heads, head_dim))?.transpose(1, 2)?;
        let v = v.reshape((1, 1, self.num_heads, head_dim))?.transpose(1, 2)?;
        
        let scale = (self.hidden_dim as f32).sqrt();
        let k_t = k.t()?;
        let scores = q.matmul(&k_t)?;
        let scores = scores.broadcast_div(&Tensor::new(scale, &scores.device())?)?;
        let attention_weights = candle_nn::ops::softmax(&scores, 3)?;
        
        // Apply attention to values
        let mut output = attention_weights.matmul(&v)?;
        
        // Reshape back to original dimensions
        // Reshape back to [batch, seq_len, hidden_dim]
        output = output.transpose(1, 2)?.reshape((1, self.hidden_dim))?;
        
        // Project back to input dimension
        let output = self.out_proj.forward(&output)?;
        
        // Convert back to Vec<f32>
        let output_vec: Vec<f32> = output.flatten_all()?.to_vec1::<f32>()?;
        
        Ok(output_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_fused_attention() -> Result<(), AttentionError> {
        let device = Device::Cpu;
        let input_dim = 64;
        let hidden_dim = 64;
        let num_heads = 8;
        let batch_size = 1;
        
        let mut attention = FusedAttention::new(
            device,
            input_dim,
            hidden_dim,
            num_heads,
            0.1,
            None,
        )?;
        
        let input = vec![1.0; input_dim * batch_size];
        let output = attention.process_mermaid_flow(input).await?;
        
        assert_eq!(output.len(), input_dim * batch_size);
        Ok(())
    }
}
