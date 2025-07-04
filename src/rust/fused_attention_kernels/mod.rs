//! Fused attention kernels for efficient transformer inference.
//! This module provides optimized attention implementations for various hardware targets.

pub mod fused_attention;
pub mod sparsity_manager;
pub mod distiller;
pub mod memory_layout;
pub mod memory_management;
pub mod speculative_decoding;
pub mod wasm;

// Re-export the main types for easier access
pub use crate::fused_attention_kernels::fused_attention::FusedAttention;
pub use crate::fused_attention_kernels::sparsity_manager::{SparsityManager, NeuralPredictor};
pub use crate::fused_attention_kernels::distiller::Distiller;
pub use crate::fused_attention_kernels::memory_layout::FFNMemoryLayout;
pub use crate::fused_attention_kernels::memory_management::MemoryManager;
pub use crate::fused_attention_kernels::speculative_decoding::SpeculativeDecoder;

// Re-export error types
pub use sparsity_manager::SparsityError;
pub use memory_layout::MemoryLayoutError;
pub use distiller::DistillerError;

use thiserror::Error;

/// Error type for attention operations
#[derive(Error, Debug)]
pub enum AttentionError {
    #[error("Attention error: {0}")]
    Generic(String),
    #[error("Failed to update KV cache")]
    KVCacheUpdate,
    #[error("Invalid input dimensions")]
    InvalidInputDims,
    #[error("Shape mismatch")]
    ShapeMismatch,
    #[error("Invalid input length")]
    InvalidInputLength,
    
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}