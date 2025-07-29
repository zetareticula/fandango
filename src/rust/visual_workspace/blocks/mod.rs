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

/// Trait that all optimization blocks must implement
#[async_trait::async_trait]
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
    async fn process(&mut self, inputs: BlockInputs) -> Result<BlockOutputs>;
    
    /// Clone the block (for duplication)
    fn clone_box(&self) -> Box<dyn OptimizationBlock>;
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInstance {
    pub id: BlockId,
    pub block_type: String,
    pub position: Position,
    pub size: Size,
    pub data: serde_json::Value,
    #[serde(skip)]
    block: Option<Box<dyn OptimizationBlock>>,
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
