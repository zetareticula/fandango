//! Fused attention kernels for efficient transformer inference.
//! This module provides optimized attention implementations for various hardware targets.

pub mod fused_attention;
pub mod sparsity_manager;
pub mod memory_layout;
pub mod cuda_triton;
pub mod distiller;
pub mod speculative_decoding;
pub mod wasm;

// Re-export the main types
pub use fused_attention::{FusedAttention, AttentionError};

// Re-export the error type for consistency
pub use crate::fused_attention_kernels::fused_attention::AttentionError as FusedAttentionError;

use candle_core::{Tensor, DType, Device};
use thiserror::Error;

/// Error type for attention operations
#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("Failed to update KV cache")]
    KVCacheUpdate,
    
    #[error("Invalid input dimensions")]
    InvalidInput,
    
    #[error("Shape mismatch in attention")]
    ShapeMismatch,
    
    #[error("Invalid input length")]
    InvalidInputLength,
    
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

impl From<FusedAttentionError> for AttentionError {
    fn from(err: FusedAttentionError) -> Self {
        match err {
            FusedAttentionError::ShapeMismatch => AttentionError::ShapeMismatch,
            FusedAttentionError::InvalidInput => AttentionError::InvalidInputLength,
            FusedAttentionError::CandleError(e) => AttentionError::CandleError(e),
            FusedAttentionError::Anyhow(e) => AttentionError::Anyhow(e),
        }
    }
}