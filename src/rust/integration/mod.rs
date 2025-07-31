//! Integration module for Fandango's core components
//! 
//! This module provides a unified interface to work with:
//! - CaaS LSM (Log-Structured Merge Tree with Compaction as a Service)
//! - Fused Attention Kernels
//! - QLoRA (Quantized Low-Rank Adaptation)
//! - IR Generator
//! - Storage Engine
//! - KV Cache Manager

pub mod core;
pub mod pipeline;
pub mod error;

// Re-export key types for easier access
pub use crate::{
    caas_lsm::{CompactionAgent, ControlPlane, CompactionRequest},
    fused_attention_kernels::{FusedAttention, SparsityManager, NeuralPredictor},
    q_lora::{QLoRAConfig, QLoRA},
    ir_generator::IRGenerator,
    storage_engine::{SelfDesigningEngine, LearnedStructure, DesignSpace},
    kvcache_manager::KVCacheManager,
};

use crate::error::Result;
use candle_core::Device;
use std::sync::Arc;

/// Main integration point for Fandango's core components
pub struct FandangoEngine {
    /// Device to run computations on (CPU/CUDA)
    device: Device,
    
    /// QLoRA component for efficient fine-tuning
    qlora: QLoRA,
    
    /// KV Cache Manager for attention caching
    kv_cache: Arc<KVCacheManager>,
    
    /// Storage engine with self-designing capabilities
    storage: Arc<SelfDesigningEngine>,
    
    /// Control plane for CaaS LSM
    control_plane: ControlPlane,
}

impl FandangoEngine {
    /// Create a new FandangoEngine instance
    pub fn new(device: Device) -> Result<Self> {
        // Initialize KV Cache Manager
        let kv_cache = Arc::new(KVCacheManager::new(device.clone()));
        
        // Initialize Storage Engine
        let storage = Arc::new(SelfDesigningEngine::new(device.clone())?);
        
        // Initialize Control Plane for CaaS LSM
        let control_plane = ControlPlane::new(kv_cache.clone(), device.clone())?;
        
        // Initialize QLoRA
        let qlora = QLoRA::new(QLoRAConfig::default(), device.clone())?;
        
        Ok(Self {
            device,
            qlora,
            kv_cache,
            storage,
            control_plane,
        })
    }
    
    /// Get a reference to the KV Cache Manager
    pub fn kv_cache(&self) -> &KVCacheManager {
        &self.kv_cache
    }
    
    /// Get a reference to the Storage Engine
    pub fn storage(&self) -> &SelfDesigningEngine {
        &self.storage
    }
    
    /// Get a reference to the QLoRA component
    pub fn qlora(&self) -> &QLoRA {
        &self.qlora
    }
    
    /// Get a reference to the Control Plane
    pub fn control_plane(&self) -> &ControlPlane {
        &self.control_plane
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_engine_initialization() -> Result<()> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let engine = FandangoEngine::new(device)?;
        
        // Verify components are initialized
        assert!(engine.kv_cache().is_ok());
        assert!(engine.storage().is_ok());
        assert!(engine.qlora().is_ok());
        
        Ok(())
    }
}
