use candle_core::{Tensor, Device, DType};
use candle_nn::{Linear, Module, VarBuilder};
use std::path::Path;
use thiserror::Error;

use crate::fused_attention_kernels::AttentionError;

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
        let vb = match weights_path {
            Some(path) => VarBuilder::from_pth(path, &device)?,
            None => VarBuilder::zeros(DType::F32, &device),
        };

        let q_proj = Linear::new(
            Tensor::zeros((input_dim, hidden_dim), DType::F32, &device)?,
            Some(Tensor::zeros((hidden_dim,), DType::F32, &device)?),
        );
        
        let k_proj = Linear::new(
            Tensor::zeros((input_dim, hidden_dim), DType::F32, &device)?,
            Some(Tensor::zeros((hidden_dim,), DType::F32, &device)?),
        );
        
        let v_proj = Linear::new(
            Tensor::zeros((input_dim, hidden_dim), DType::F32, &device)?,
            Some(Tensor::zeros((hidden_dim,), DType::F32, &device)?),
        );
        
        let out_proj = Linear::new(
            Tensor::zeros((hidden_dim, input_dim), DType::F32, &device)?,
            Some(Tensor::zeros((input_dim,), DType::F32, &device)?),
        );

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

    pub async fn process_mermaid_flow(&mut self, input: Vec<f32>) -> Result<Vec<f32>, AttentionError> {
        // Convert input to tensor
        let input_len = input.len();
        if input_len % self.input_dim != 0 {
            return Err(AttentionError::InvalidInput);
        }
        
        let batch_size = input_len / self.input_dim;
        let input_tensor = Tensor::from_vec(
            input,
            (batch_size, self.input_dim),
            &self.device,
        )?;

        // Project input to query, key, value
        let q = self.q_proj.forward(&input_tensor)?;
        let k = self.k_proj.forward(&input_tensor)?;
        let v = self.v_proj.forward(&input_tensor)?;

        // Reshape for multi-head attention
        let batch_size = q.dim(0)?;
        let head_dim = self.hidden_dim / self.num_heads;
        
        let q = q.reshape((batch_size, -1, self.num_heads, head_dim))?;
        let k = k.reshape((batch_size, -1, self.num_heads, head_dim))?;
        let v = v.reshape((batch_size, -1, self.num_heads, head_dim))?;

        // Scaled dot-product attention
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scale = (head_dim as f64).sqrt() as f32;
        let scores = scores.broadcast_div(&Tensor::new(scale, &self.device)?)?;
        let attention_weights = scores.softmax(3)?;
        
        // Apply attention to values
        let mut output = attention_weights.matmul(&v)?;
        
        // Reshape back to original dimensions
        output = output.transpose(1, 2)?.contiguous()?;
        output = output.reshape((batch_size, -1))?;
        
        // Final projection
        output = self.out_proj.forward(&output)?;
        
        // Convert back to Vec<f32>
        output.to_vec1::<f32>()
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
