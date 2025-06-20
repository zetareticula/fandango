use crate::runtime_scheduler::token_window::TokenWindow;
use crate::runtime_scheduler::model_loader::ModelLoader;
use std::collections::HashMap;
// src/rust/runtime_scheduler/mod.rs

// RuntimeScheduler for managing model loading and scheduling based on token windows
pub struct RuntimeScheduler {
    token_window: TokenWindow,
    model_loader: ModelLoader,
    cached_layers: std::collections::HashMap<String, Vec<u8>>,
}

impl RuntimeScheduler {
    pub fn new() -> Self {
        RuntimeScheduler {
            token_window: TokenWindow::new(),
            model_loader: ModelLoader::new(),
            cached_layers: HashMap::new(),
        }
    }

    pub fn schedule(&mut self, model_id: &str, tokens: usize, sla: f32) {
        let load_time = self.token_window.estimate_load(tokens, sla);
        if load_time < 10.0 { // ms threshold for on-demand loading
            let quantized_layer = self.model_loader.load_quantized(model_id);
            self.cached_layers.insert(model_id.to_string(), quantized_layer);
        } else {
            self.model_loader.unload(model_id);
            self.cached_layers.remove(model_id);
        }
    }
}


impl Drop for RuntimeScheduler {
    fn drop(&mut self) {
        for (model_id, _) in &self.cached_layers {
            self.model_loader.unload(model_id);
        }
    }
}

// Example usage
// fn main() {
//     let mut scheduler = RuntimeScheduler::new();
//     scheduler.schedule("example_model", 1000, 0.5);
//     // Simulate some operations
//     std::thread::sleep(std::time::Duration::from_secs(1));
//     // Scheduler will automatically unload models on drop
// }
// This code defines a `RuntimeScheduler` that manages model loading and scheduling based on token windows.
// It uses a `TokenWindow` to estimate load times and a `ModelLoader` to handle quantized model loading.
