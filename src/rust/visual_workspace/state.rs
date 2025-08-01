
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

//! Workspace state management
//! Handles the current state of the visual workspace

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use super::{BlockId, Position, Result, WorkspaceError};
use crate::visual_workspace::blocks::{BlockInstance, BlockInputs, BlockOutputs};

/// Connection between two blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub from_block: BlockId,
    pub from_port: String,
    pub to_block: BlockId,
    pub to_port: String,
}

/// The complete state of the workspace
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct WorkspaceState {
    /// All blocks in the workspace
    pub blocks: HashMap<BlockId, BlockInstance>,
    
    /// Connections between blocks
    pub connections: Vec<Connection>,
    
    /// Viewport position
    pub viewport: Viewport,
    
    /// Zoom level
    pub zoom: f32,
    
    /// Next Z-index for block ordering
    next_z_index: u32,
}

/// Viewport information
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1024.0,
            height: 768.0,
        }
    }
}

impl WorkspaceState {
    /// Create a new empty workspace
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a new block to the workspace
    pub fn add_block(&mut self, mut block: BlockInstance) -> BlockId {
        let id = block.id;
        self.blocks.insert(id, block);
        self.bring_to_front(id);
        id
    }
    
    /// Remove a block and all its connections
    pub fn remove_block(&mut self, block_id: BlockId) -> Result<()> {
        // Remove the block
        self.blocks.remove(&block_id).ok_or_else(|| 
            WorkspaceError::StateError("Block not found".to_string())
        )?;
        
        // Remove any connections involving this block
        self.connections.retain(|conn| 
            conn.from_block != block_id && conn.to_block != block_id
        );
        
        Ok(())
    }
    
    /// Add a connection between two blocks
    pub fn add_connection(
        &mut self,
        from_block: BlockId,
        from_port: String,
        to_block: BlockId,
        to_port: String,
    ) -> Result<()> {
        // Validate blocks exist
        if !self.blocks.contains_key(&from_block) || !self.blocks.contains_key(&to_block) {
            return Err(WorkspaceError::StateError("Invalid block IDs".to_string()));
        }
        
        // TODO: Validate port types match
        
        let connection = Connection {
            from_block,
            from_port,
            to_block,
            to_port,
        };
        
        self.connections.push(connection);
        Ok(())
    }
    
    /// Remove a connection
    pub fn remove_connection(&mut self, connection_index: usize) -> Result<()> {
        if connection_index >= self.connections.len() {
            return Err(WorkspaceError::StateError("Invalid connection index".to_string()));
        }
        
        self.connections.remove(connection_index);
        Ok(())
    }
    
    /// Bring a block to the front
    pub fn bring_to_front(&mut self, block_id: BlockId) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            self.next_z_index += 1;
            // Update z-index
            // (implementation depends on your rendering system)
        }
    }
    
    /// Get all connections for a block
    pub fn get_block_connections(&self, block_id: BlockId) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|conn| conn.from_block == block_id || conn.to_block == block_id)
            .collect()
    }
    
    /// Execute the pipeline
    pub async fn execute(&self) -> Result<HashMap<BlockId, BlockOutputs>> {
        // TODO: Implement topological sort and execution
        // This would involve:
        // 1. Building a dependency graph
        // 2. Topologically sorting the blocks
        // 3. Executing blocks in order
        // 4. Passing outputs to connected blocks
        
        Ok(HashMap::new())
    }
    
    /// Serialize the workspace state to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(WorkspaceError::from)
    }
    
    /// Deserialize the workspace state from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(WorkspaceError::from)
    }
}
