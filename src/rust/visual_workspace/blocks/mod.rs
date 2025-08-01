
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

//! Block system for the visual workspace
//! Defines the base block trait and common block implementations

use std::sync::Arc;
use serde::{Serialize, Deserialize};
use super::{Position, Size, BlockId, Result, WorkspaceError};

/// Trait for debugging information that blocks can provide
pub trait Debuggable {
    /// Get debug information as a string
    fn debug_info(&self) -> String;
    
    /// Get detailed metrics as a JSON value
    fn metrics(&self) -> Option<serde_json::Value> {
        None
    }
    
    /// Get the current progress (0.0 to 1.0)
    fn progress(&self) -> f32 {
        0.0
    }
}

/// Extension trait for blocks that support debugging
pub trait DebuggableBlock: OptimizationBlock + Debuggable {}

// Implement Debuggable for all blocks that implement both OptimizationBlock and Debuggable
impl<T> DebuggableBlock for T where T: OptimizationBlock + Debuggable {}
use crate::fused_attention_kernels::fused_attention::FusedAttention;
use crate::fused_attention_kernels::sparsity_manager::{SparsityManager, NeuralPredictor};
use crate::fused_attention_kernels::distiller::Distiller;
use crate::fused_attention_kernels::memory_layout::FFNMemoryLayout;
use crate::fused_attention_kernels::memory_management::MemoryManager;
use crate::fused_attention_kernels::speculative_decoding::SpeculativeDecoder;
use crate::fused_attention_kernels::wasm::WasmFusedAttention;
use crate::kvcache_manager::KVCacheManager;
use crate::runtime_scheduler::RuntimeScheduler;
use crate::utils::Timer;
use crate::storage_engine::SelfDesigningEngine;
use crate::cognitive_modeling::MCMCSearch;

use std::future::Future;
use std::pin::Pin;

/// Trait that all optimization blocks must implement
pub trait OptimizationBlock: Send + Sync + std::fmt::Debug {
    /// Unique identifier for this block type
    fn block_type(&self) -> &'static str;
    
    /// Human-readable name
    fn name(&self) -> &'static str;
    
    /// Description of what this block does
    fn description(&self) -> &'static str;
    
    /// Category for the block library
    fn category(&self) -> &'static str;
    
    /// Input ports (name, type)
    fn inputs(&self) -> Vec<(&'static str, &'static str)>;
    
    /// Output ports (name, type)
    fn outputs(&self) -> Vec<(&'static str, &'static str)>;
    
    /// Process the input data and return the output
    fn process<'a>(
        &'a mut self,
        inputs: BlockInputs,
    ) -> Pin<Box<dyn Future<Output = Result<BlockOutputs>> + Send + 'a>>;
    
    /// Clone the block (for duplication)
    fn clone_box(&self) -> Box<dyn OptimizationBlock>;
    
    /// Get a reference to the block as Any
    fn as_any(&self) -> &dyn std::any::Any;
    
    /// Get a mutable reference to the block as Any
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// Input values for a block
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BlockInputs {
    pub values: std::collections::HashMap<String, serde_json::Value>,
}

/// Output values from a block
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BlockOutputs {
    pub values: std::collections::HashMap<String, serde_json::Value>,
}

/// A block instance in the workspace
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockInstance {
    pub id: BlockId,
    pub block_type: String,
    pub position: Position,
    pub size: Size,
    pub data: serde_json::Value,
    #[serde(skip)]
    block: Option<Box<dyn OptimizationBlock>>,
}

impl Clone for BlockInstance {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            block_type: self.block_type.clone(),
            position: self.position.clone(),
            size: self.size.clone(),
            data: self.data.clone(),
            block: self.block.as_ref().map(|b| b.clone_box()),
        }
    }
}

impl BlockInstance {
    /// Create a new block instance
    pub fn new(block: Box<dyn OptimizationBlock>) -> Self {
        Self {
            id: BlockId::default(),
            block_type: block.block_type().to_string(),
            position: Position::default(),
            size: Size { width: 200.0, height: 100.0 },
            data: serde_json::Value::Null,
            block: Some(block),
        }
    }
    
    /// Get a reference to the underlying block
    pub fn block(&self) -> Result<&dyn OptimizationBlock> {
        self.block.as_deref().ok_or_else(|| 
            WorkspaceError::BlockError("Block not initialized".to_string())
        )
    }
    
    /// Get a mutable reference to the underlying block as Any
    pub fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        // If we have a block, return its Any, otherwise return self
        // We need to handle the Option properly to avoid multiple mutable borrows
        match self.block {
            Some(ref mut block) => block.as_any_mut(),
            None => self,
        }
    }
    
    /// Get a reference to the block as Any
    pub fn as_any(&self) -> &dyn std::any::Any {
        self.block.as_ref().map(|b| b.as_any()).unwrap_or(self)
    }
    
    /// Get a mutable reference to the underlying block
    pub fn block_mut(&mut self) -> Result<&mut Box<dyn OptimizationBlock>> {
        self.block.as_mut().ok_or_else(|| 
            WorkspaceError::BlockError("Block not initialized".to_string())
        )
    }
}

/// Library of available blocks
#[derive(Debug, Default)]
pub struct BlockLibrary {
    blocks: std::collections::HashMap<String, Box<dyn OptimizationBlock>>,
}

impl BlockLibrary {
    /// Create a new block library with default blocks
    pub fn new() -> Self {
        let mut library = Self::default();
        // Register default blocks
        library.register(Box::new(QuantizationBlock::new(4)));
        library.register(Box::new(SparseAttentionBlock::default()));
        library.register(Box::new(KVCacheOptimizer::default()));
        library
    }
    
    /// Register a new block type
    pub fn register(&mut self, block: Box<dyn OptimizationBlock>) {
        self.blocks.insert(block.block_type().to_string(), block);
    }
    
    /// Create a new instance of a block
    pub fn create_instance(&self, block_type: &str) -> Result<BlockInstance> {
        let block = self.blocks.get(block_type)
            .ok_or_else(|| 
                WorkspaceError::BlockError(format!("Unknown block type: {}", block_type))
            )?
            .clone_box();
            
        Ok(BlockInstance::new(block))
    }
    
    /// Get all available block types grouped by category
    pub fn blocks_by_category(&self) -> std::collections::HashMap<&'static str, Vec<&'static str>> {
        let mut categories = std::collections::HashMap::new();
        
        for block in self.blocks.values() {
            categories
                .entry(block.category())
                .or_insert_with(Vec::new)
                .push(block.name());
        }
        
        categories
    }
}

// Block implementations
mod quantization;
mod sparse_attention;
mod kv_cache;
mod mcmc;

pub use quantization::QuantizationBlock;
pub use sparse_attention::SparseAttentionBlock;
pub use kv_cache::KVCacheOptimizer;
pub use mcmc::MCMCBlock;
