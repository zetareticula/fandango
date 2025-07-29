//! Optimization pipeline execution
//! Handles the execution of the optimization pipeline defined in the workspace

use std::collections::{HashMap, VecDeque};
use super::{
    BlockId, Result, WorkspaceError,
    blocks::{BlockInstance, BlockInputs, BlockOutputs},
    state::{WorkspaceState, Connection},
};

/// Represents a node in the execution graph
struct ExecutionNode {
    block_id: BlockId,
    block: BlockInstance,
    dependencies: Vec<BlockId>,
    dependents: Vec<BlockId>,
    input_connections: Vec<Connection>,
    output_connections: Vec<Connection>,
}

/// The optimization pipeline executor
pub struct OptimizationPipeline {
    nodes: HashMap<BlockId, ExecutionNode>,
    ready_queue: VecDeque<BlockId>,
    in_progress: HashMap<BlockId, tokio::task::JoinHandle<Result<BlockOutputs>>>,
    completed: HashMap<BlockId, BlockOutputs>,
}

impl OptimizationPipeline {
    /// Create a new pipeline from the workspace state
    pub fn new(state: &WorkspaceState) -> Result<Self> {
        let mut nodes = HashMap::new();
        
        // Create execution nodes for each block
        for (&block_id, block) in &state.blocks {
            let node = ExecutionNode {
                block_id,
                block: block.clone(),
                dependencies: Vec::new(),
                dependents: Vec::new(),
                input_connections: Vec::new(),
                output_connections: Vec::new(),
            };
            nodes.insert(block_id, node);
        }
        
        // Process connections to build the dependency graph
        for conn in &state.connections {
            if let Some(node) = nodes.get_mut(&conn.to_block) {
                node.input_connections.push(conn.clone());
                node.dependencies.push(conn.from_block);
            }
            
            if let Some(node) = nodes.get_mut(&conn.from_block) {
                node.output_connections.push(conn.clone());
                node.dependents.push(conn.to_block);
            }
        }
        
        // Find initial nodes (no dependencies)
        let ready_queue = nodes
            .values()
            .filter(|node| node.dependencies.is_empty())
            .map(|node| node.block_id)
            .collect();
        
        Ok(Self {
            nodes,
            ready_queue,
            in_progress: HashMap::new(),
            completed: HashMap::new(),
        })
    }
    
    /// Execute the pipeline with a specified level of parallelism
    pub async fn execute(mut self) -> Result<HashMap<BlockId, BlockOutputs>> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| WorkspaceError::PipelineError(e.to_string()))?;
        
        runtime.block_on(async move {
            let max_concurrent = 4; // Maximum concurrent blocks to execute
            
            while !self.ready_queue.is_empty() || !self.in_progress.is_empty() {
                // Start new tasks if we have capacity
                while self.in_progress.len() < max_concurrent && !self.ready_queue.is_empty() {
                    if let Some(block_id) = self.ready_queue.pop_front() {
                        self.start_block(block_id).await?;
                    }
                }
                
                // Wait for at least one task to complete
                if !self.in_progress.is_empty() {
                    self.await_next_completion().await?;
                }
            }
            
            Ok(self.completed)
        })
    }
    
    /// Start executing a block
    async fn start_block(&mut self, block_id: BlockId) -> Result<()> {
        let node = self.nodes.get_mut(&block_id)
            .ok_or_else(|| WorkspaceError::PipelineError("Block not found".to_string()))?;
        
        // Gather inputs from connected blocks
        let mut inputs = BlockInputs::default();
        
        for conn in &node.input_connections {
            if let Some(outputs) = self.completed.get(&conn.from_block) {
                if let Some(value) = outputs.values.get(&conn.from_port) {
                    inputs.values.insert(conn.to_port.clone(), value.clone());
                }
            }
        }
        
        // Clone the block for execution
        let block = node.block.clone();
        
        // Spawn a new task to execute the block
        let handle = tokio::spawn(async move {
            let mut block = block;
            block.block_mut()?.process(inputs).await
        });
        
        self.in_progress.insert(block_id, handle);
        Ok(())
    }
    
    /// Wait for the next block to complete and update the state
    async fn await_next_completion(&mut self) -> Result<()> {
        // Wait for any task to complete
        let (block_id, result) = {
            let (block_id, handle) = self.in_progress.iter_mut().next()
                .ok_or_else(|| WorkspaceError::PipelineError("No tasks in progress".to_string()))?;
            
            let block_id = *block_id;
            let result = handle.await
                .map_err(|e| WorkspaceError::PipelineError(format!("Task panicked: {}", e)))?;
                
            (block_id, result)
        };
        
        // Remove the completed task
        self.in_progress.remove(&block_id);
        
        // Handle the result
        let outputs = result?;
        self.completed.insert(block_id, outputs);
        
        // Find any blocks that are now ready to execute
        if let Some(node) = self.nodes.get(&block_id) {
            for &dependent_id in &node.dependents {
                if let Some(dep_node) = self.nodes.get_mut(&dependent_id) {
                    // Check if all dependencies are satisfied
                    if dep_node.dependencies.iter().all(|id| self.completed.contains_key(id)) {
                        self.ready_queue.push_back(dependent_id);
                    }
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual_workspace::blocks::BlockLibrary;
    
    #[tokio::test]
    async fn test_pipeline_execution() {
        // TODO: Add test cases for pipeline execution
    }
}
