use crate::runtime_scheduler::token_window::TokenWindow;
use crate::runtime_scheduler::model_loader::ModelLoader;

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