//! Visual workspace implementation
//! Handles the main workspace UI and interaction

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};
use super::{
    BlockId, Position, Result, Size, WorkspaceError,
    blocks::{BlockInstance, BlockLibrary, Debuggable, DebuggableBlock},
    state::{WorkspaceState, Connection},
    pipeline::OptimizationPipeline,
};

/// Represents the execution status of the workspace
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// No execution is currently running
    Idle,
    
    /// Execution is currently running
    Running,
    
    /// Execution is paused
    Paused,
}

/// Represents the current execution state of the workspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// Current execution status
    pub status: ExecutionStatus,
    
    /// Current progress (0.0 to 1.0)
    pub progress: f32,
    
    /// Additional execution metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
}

/// The main visual workspace
pub struct VisualWorkspace {
    /// Current workspace state
    pub state: WorkspaceState,
    
    /// Block library
    pub block_library: BlockLibrary,
    
    /// Selected blocks
    selected_blocks: Vec<BlockId>,
    
    /// Currently dragged block (if any)
    dragged_block: Option<(BlockId, Position)>,
    
    /// Currently dragged connection (if any)
    dragged_connection: Option<DragConnection>,
    
    /// Execution pipeline
    pipeline: Option<OptimizationPipeline>,
    
    /// Debug information for blocks
    debug_info: Arc<Mutex<HashMap<BlockId, String>>>,
    
    /// Execution metrics
    metrics: Arc<Mutex<HashMap<BlockId, serde_json::Value>>>,
    
    /// Execution progress (0.0 to 1.0)
    progress: Arc<Mutex<f32>>,
    
    /// Is execution currently running?
    is_running: bool,
    
    /// Should execution be paused?
    should_pause: bool,
    
    /// Should execution be stopped?
    should_stop: bool,
}

/// Represents a connection being dragged
struct DragConnection {
    from_block: BlockId,
    from_port: String,
    position: Position,
}

impl Default for VisualWorkspace {
    fn default() -> Self {
        Self {
            state: WorkspaceState::new(),
            block_library: BlockLibrary::new(),
            selected_blocks: Vec::new(),
            dragged_block: None,
            dragged_connection: None,
            pipeline: None,
            debug_info: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(HashMap::new())),
            progress: Arc::new(Mutex::new(0.0)),
            is_running: false,
            should_pause: false,
            should_stop: false,
        }
    }
}

impl VisualWorkspace {
    /// Create a new visual workspace
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add a new block to the workspace
    pub fn add_block(&mut self, block_type: &str, position: Position) -> Result<BlockId> {
        let mut block = self.block_library.create_instance(block_type)?;
        block.position = position;
        let id = block.id;
        self.state.add_block(block);
        Ok(id)
    }
    
    /// Remove selected blocks
    pub fn remove_selected_blocks(&mut self) -> Result<()> {
        for &block_id in &self.selected_blocks {
            self.state.remove_block(block_id)?;
        }
        self.selected_blocks.clear();
        Ok(())
    }
    
    /// Start dragging a block
    pub fn start_dragging_block(&mut self, block_id: BlockId, mouse_pos: Position) -> Result<()> {
        if let Some(block) = self.state.blocks.get(&block_id) {
            self.dragged_block = Some((block_id, Position {
                x: mouse_pos.x - block.position.x,
                y: mouse_pos.y - block.position.y,
            }));
            self.select_block(block_id, false);
            Ok(())
        } else {
            Err(WorkspaceError::StateError("Block not found".to_string()))
        }
    }
    
    /// Update block position during drag
    pub fn update_dragged_block(&mut self, mouse_pos: Position) -> Result<()> {
        if let Some((block_id, offset)) = &self.dragged_block {
            if let Some(block) = self.state.blocks.get_mut(block_id) {
                block.position = Position {
                    x: mouse_pos.x - offset.x,
                    y: mouse_pos.y - offset.y,
                };
                return Ok(());
            }
        }
        Err(WorkspaceError::StateError("No block being dragged".to_string()))
    }
    
    /// Stop dragging a block
    pub fn stop_dragging_block(&mut self) {
        self.dragged_block = None;
    }
    
    /// Start dragging a connection
    pub fn start_dragging_connection(
        &mut self,
        block_id: BlockId,
        port_name: String,
        position: Position,
    ) -> Result<()> {
        if !self.state.blocks.contains_key(&block_id) {
            return Err(WorkspaceError::StateError("Block not found".to_string()));
        }
        
        self.dragged_connection = Some(DragConnection {
            from_block: block_id,
            from_port: port_name,
            position,
        });
        
        Ok(())
    }
    
    /// Update connection position during drag
    pub fn update_dragged_connection(&mut self, position: Position) {
        if let Some(connection) = &mut self.dragged_connection {
            connection.position = position;
        }
    }
    
    /// Complete a connection
    pub fn complete_connection(
        &mut self,
        to_block: BlockId,
        to_port: String,
    ) -> Result<()> {
        if let Some(drag) = self.dragged_connection.take() {
            self.state.add_connection(
                drag.from_block,
                drag.from_port,
                to_block,
                to_port,
            )?;
        }
        Ok(())
    }
    
    /// Cancel the current drag operation
    pub fn cancel_drag(&mut self) {
        self.dragged_block = None;
        self.dragged_connection = None;
    }
    
    /// Select a block (add to selection if shift is pressed)
    pub fn select_block(&mut self, block_id: BlockId, multi_select: bool) {
        if !multi_select {
            self.selected_blocks.clear();
        }
        
        if !self.selected_blocks.contains(&block_id) {
            self.selected_blocks.push(block_id);
        }
        
        self.state.bring_to_front(block_id);
    }
    
    /// Clear the current selection
    pub fn clear_selection(&mut self) {
        self.selected_blocks.clear();
    }
    
    /// Get the current selection
    pub fn selection(&self) -> &[BlockId] {
        &self.selected_blocks
    }
    
    /// Get the block being dragged (if any)
    pub fn dragged_block(&self) -> Option<(BlockId, Position)> {
        self.dragged_block
            .as_ref()
            .map(|(id, offset)| (*id, *offset))
    }
    
    /// Get the connection being dragged (if any)
    pub fn dragged_connection(&self) -> Option<(&BlockId, &str, &Position)> {
        self.dragged_connection
            .as_ref()
            .map(|drag| (&drag.from_block, drag.from_port.as_str(), &drag.position))
    }
    
    /// Execute the pipeline with iterative debugging
    pub async fn execute_iterative(&mut self) -> Result<()> {
        if self.is_running {
            return Err(WorkspaceError::StateError("Execution already in progress".to_string()));
        }
        
        self.is_running = true;
        self.should_pause = false;
        self.should_stop = false;
        *self.progress.lock().unwrap() = 0.0;
        self.debug_info.lock().unwrap().clear();
        self.metrics.lock().unwrap().clear();
        
        // Create a new pipeline
        let pipeline = OptimizationPipeline::new(&self.state)?;
        self.pipeline = Some(pipeline);
        
        // Start execution in a separate task
        let debug_info = Arc::clone(&self.debug_info);
        let metrics = Arc::clone(&self.metrics);
        let progress = Arc::clone(&self.progress);
        let should_pause = Arc::new(Mutex::new(false));
        let should_stop = Arc::new(Mutex::new(false));
        
        // Clone the pipeline for the background task
        let mut pipeline = self.pipeline.take().unwrap();
        
        tokio::spawn(async move {
            // Run the pipeline with periodic updates
            let result = pipeline.execute_with_callbacks(
                |block_id, block| {
                    // Update debug info
                    if let Some(debuggable) = block.as_any().downcast_ref::<Box<dyn Debuggable>>() {
                        debug_info.lock().unwrap()
                            .insert(block_id, debuggable.debug_info());
                            
                        if let Some(block_metrics) = debuggable.metrics() {
                            metrics.lock().unwrap()
                                .insert(block_id, block_metrics);
                        }
                    }
                    
                    // Update progress
                    *progress.lock().unwrap() = block_id.0 as f32 / 100.0; // Simple progress estimation
                    
                    // Check for pause/stop
                    let should_pause = *should_pause.lock().unwrap();
                    let should_stop = *should_stop.lock().unwrap();
                    
                    if should_stop {
                        return false; // Stop execution
                    }
                    
                    if should_pause {
                        // Wait until unpaused
                        while *should_pause.lock().unwrap() {
                            std::thread::sleep(std::time::Duration::from_millis(100));
                            
                            if *should_stop.lock().unwrap() {
                                return false; // Stop if requested while paused
                            }
                        }
                    }
                    
                    true // Continue execution
                }
            ).await;
            
            match result {
                Ok(_) => {
                    *progress.lock().unwrap() = 1.0;
                    // TODO: Notify UI of completion
                },
                Err(e) => {
                    // TODO: Notify UI of error
                    eprintln!("Pipeline execution error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Pause the current execution
    pub fn pause_execution(&mut self) -> Result<()> {
        if !self.is_running {
            return Err(WorkspaceError::StateError("No execution in progress".to_string()));
        }
        
        self.should_pause = true;
        Ok(())
    }
    
    /// Resume a paused execution
    pub fn resume_execution(&mut self) -> Result<()> {
        if !self.is_running {
            return Err(WorkspaceError::StateError("No execution in progress".to_string()));
        }
        
        self.should_pause = false;
        Ok(())
    }
    
    /// Stop the current execution
    pub fn stop_execution(&mut self) -> Result<()> {
        if !self.is_running {
            return Err(WorkspaceError::StateError("No execution in progress".to_string()));
        }
        
        self.should_stop = true;
        self.is_running = false;
        Ok(())
    }
    
    /// Get the current execution status
    pub fn execution_status(&self) -> ExecutionStatus {
        if !self.is_running {
            return ExecutionStatus::Idle;
        }
        
        if self.should_pause {
            ExecutionStatus::Paused
        } else {
            ExecutionStatus::Running
        }
    }
    
    /// Get debug information for a block
    pub fn get_debug_info(&self, block_id: BlockId) -> Option<String> {
        self.debug_info.lock().unwrap().get(&block_id).cloned()
    }
    
    /// Get metrics for a block
    pub fn get_metrics(&self, block_id: BlockId) -> Option<serde_json::Value> {
        self.metrics.lock().unwrap().get(&block_id).cloned()
    }
    
    /// Get the current execution progress (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        *self.progress.lock().unwrap()
    }
    
    /// Get the current execution state
    pub fn execution_state(&self) -> ExecutionState {
        ExecutionState {
            status: self.execution_status(),
            progress: self.progress(),
            metrics: None, // We can add metrics collection here if needed
        }
    }
    
    /// Serialize the workspace to JSON
    pub fn to_json(&self) -> Result<String> {
        self.state.to_json()
    }
    
    /// Deserialize the workspace from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        let state = WorkspaceState::from_json(json)?;
        Ok(Self {
            state,
            block_library: BlockLibrary::new(),
            selected_blocks: Vec::new(),
            dragged_block: None,
            dragged_connection: None,
        })
    }
}
