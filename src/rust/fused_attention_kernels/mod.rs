pub mod fused_attention;
pub mod sparsity_manager;
pub mod memory_layout;
pub mod cuda_triton;
pub mod distiller;
pub mod speculative_decoding;
pub mod wasm;

// Re-export the main FusedAttention struct
pub use fused_attention::FusedAttention;

use candle_core::{Tensor, DType, Device};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Candle error: {0}")]
    CandleError(#[from] candle_core::Error),
    
    #[error("Failed to update KV cache")]
    KVCacheUpdate,
    
    #[error("Invalid input dimensions")]
    InvalidInput,
}