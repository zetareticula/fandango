
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

//! Visual workspace for drag-and-drop LLM optimization
//! Provides an intuitive interface for building optimization pipelines

// Define types first
use uuid;

/// Position in the 2D workspace
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Default for Position {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

/// Size of a block in the workspace
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

/// Unique identifier for blocks in the workspace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BlockId(uuid::Uuid);

impl BlockId {
    /// Creates a new unique BlockId
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
    
    /// Returns the underlying UUID
    pub fn as_uuid(&self) -> &uuid::Uuid {
        &self.0
    }
}

impl Default for BlockId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_string().chars().take(8).collect::<String>())
    }
}

impl From<uuid::Uuid> for BlockId {
    fn from(uuid: uuid::Uuid) -> Self {
        Self(uuid)
    }
}

impl From<BlockId> for uuid::Uuid {
    fn from(block_id: BlockId) -> Self {
        block_id.0
    }
}

// Define submodules after the types they depend on
pub mod blocks;
pub mod pipeline;
pub mod state;
pub mod workspace;

// Re-exports for external use
// Note: These are used by other crates, so we keep them even if they appear unused in this file

/// Common result type for workspace operations
pub type Result<T> = std::result::Result<T, WorkspaceError>;

/// Error type for workspace operations
#[derive(thiserror::Error, Debug)]
pub enum WorkspaceError {
    #[error("Block error: {0}")]
    BlockError(String),
    
    #[error("Pipeline error: {0}")]
    PipelineError(String),
    
    #[error("State error: {0}")]
    StateError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Block not found: {0:?}")]
    BlockNotFound(BlockId),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Candle core error: {0}")]
    CandleError(#[from] candle_core::Error),
}
