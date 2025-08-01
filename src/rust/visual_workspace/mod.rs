
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

//! Visual workspace for drag-and-drop LLM optimization
//! Provides an intuitive interface for building optimization pipelines

pub mod blocks;
mod pipeline;
mod state;
mod workspace;

// Re-exports
pub use pipeline::OptimizationPipeline;
pub use state::WorkspaceState;
pub use workspace::VisualWorkspace;
pub use blocks::BlockLibrary;

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
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

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
pub struct BlockId(uuid::Uuid);

impl Default for BlockId {
    fn default() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_string().chars().take(8).collect::<String>())
    }
}
