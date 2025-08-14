//! Optimization pipeline execution
//! Handles the execution of the optimization pipeline defined in the workspace

use std::collections::{HashMap, VecDeque};
use crate::{
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
    
    /// Execute the pipeline with progress and completion callbacks
    pub async fn execute_with_callbacks<F, G>(
        self,
        on_progress: F,
        on_complete: G,
    ) -> Result<HashMap<BlockId, BlockOutputs>>
    where
        F: Fn(BlockId, &BlockInstance) + Send + 'static,
        G: Fn(BlockId, &BlockInstance, Result<BlockOutputs>) + Send + 'static + Clone,
    {
        // Create a channel for progress updates
        let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Create a channel for completion updates
        let (completion_tx, mut completion_rx) = tokio::sync::mpsc::unbounded_channel();

        // Clone the callbacks for use in the spawned task
        let progress_cb = {
            let progress_tx = progress_tx.clone();
            move |block_id: BlockId, block: &BlockInstance| {
                let _ = progress_tx.send((block_id, block.clone()));
            }
        };
        
        // Create a separate sender for the completion callback
        let completion_tx_clone = completion_tx.clone();
        let _completion_cb = move |block_id: BlockId, block: &BlockInstance, result: Result<BlockOutputs>| {
            let _ = completion_tx_clone.send((block_id, block.clone(), result));
        };
        
        // Clone the on_complete callback before moving it into the closure
        let on_complete_clone = on_complete.clone();
        let on_complete_wrapper = move |block_id: BlockId, block: &BlockInstance, result: Result<BlockOutputs>| {
            on_complete_clone(block_id, block, result);
        };

        // Spawn a task to process progress and completion updates
        let progress_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    Some((block_id, block)) = progress_rx.recv() => {
                        on_progress(block_id, &block);
                    },
                    Some((block_id, block, result)) = completion_rx.recv() => {
                        // Forward the result directly since it's already in the correct format
                        on_complete(block_id, &block, result);
                    },
                    else => break,
                }
            }
        });

        // Execute the pipeline with progress and completion callbacks
        let result = self.execute_with_progress(progress_cb, on_complete_wrapper).await;

        // Wait for all updates to be processed
        drop(progress_tx);
        drop(completion_tx);
        
        // Handle the progress handle result
        if let Err(e) = progress_handle.await {
            return Err(WorkspaceError::PipelineError(e.to_string()));
        }

        result
    }

    /// Execute the pipeline with progress updates
    pub async fn execute_with_progress<F, G>(
        mut self,
        on_progress: F,
        on_complete: G,
    ) -> Result<HashMap<BlockId, BlockOutputs>>
    where
        F: Fn(BlockId, &BlockInstance) + Send + 'static,
        G: Fn(BlockId, &BlockInstance, Result<BlockOutputs>) + Send + 'static,
    {
        // Execute the pipeline with progress updates
        let mut results: HashMap<BlockId, BlockOutputs> = HashMap::new();
        
        while let Some(block_id) = self.ready_queue.pop_front() {
            let node = match self.nodes.get(&block_id) {
                Some(node) => node,
                None => return Err(WorkspaceError::BlockNotFound(block_id)),
            };
            
            // Call progress callback with the block instance
            on_progress(block_id, &node.block);
            
            // Execute the block (simplified)
            let outputs = BlockOutputs::new(); // Replace with actual execution
            
            // Store the outputs in the completed map
            self.completed.insert(block_id, outputs.clone());
            
            // Call completion callback with the block instance and outputs
            on_complete(block_id, &node.block, Ok(outputs));
            
            // Process dependents
            let mut ready_dependents = Vec::new();
            for &dep_id in &node.dependents {
                if let Some(dep_node) = self.nodes.get(&dep_id) {
                    if dep_node.dependencies.iter().all(|id| self.completed.contains_key(id)) {
                        ready_dependents.push(dep_id);
                    }
                }
            }
            
            // Add ready dependents to the queue
            for dep_id in ready_dependents {
                self.ready_queue.push_back(dep_id);
            }
            
            // Store the results
            if let Some(outputs) = self.completed.get(&block_id) {
                results.insert(block_id, outputs.clone());
            }
        }
        
        Ok(results)
    }

    /// Execute the pipeline with a specified level of parallelism
    pub async fn execute(self) -> Result<HashMap<BlockId, BlockOutputs>> {
        // Create a no-op progress callback
        let progress_cb = |_block_id: BlockId, _block: &BlockInstance| {
            // No-op progress callback
        };
        
        // Create a no-op completion callback
        let completion_cb = |_block_id: BlockId, block: &BlockInstance, result: Result<BlockOutputs>| {
            match result {
                Ok(_) => log::info!("Completed block {}: {}", _block_id, block.block_type),
                Err(e) => log::error!("Error in block {}: {}", _block_id, e),
            }
        };

        // Execute with callbacks
        self.execute_with_progress(progress_cb, completion_cb).await
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
            block.process(inputs).await
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
        let ready_dependents: Vec<BlockId> = {
            let node = match self.nodes.get(&block_id) {
                Some(node) => node,
                None => return Ok(()),
            };
            
            node.dependents.iter()
                .filter(|&&dep_id| {
                    // Check if all dependencies are satisfied
                    self.nodes.get(&dep_id).map_or(false, |dep_node| {
                        dep_node.dependencies.iter().all(|id| self.completed.contains_key(id))
                    })
                })
                .cloned()
                .collect()
        };
        
        // Add ready dependents to the queue
        for dep_id in ready_dependents {
            self.ready_queue.push_back(dep_id);
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visual_workspace::blocks::{Block, BlockLibrary};
    use crate::visual_workspace::state::WorkspaceState;
    
    #[tokio::test]
    async fn test_pipeline_execution() {
        // TODO: Add test cases for pipeline execution
    }
}
