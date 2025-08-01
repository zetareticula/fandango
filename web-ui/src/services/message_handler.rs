use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;
use std::collections::HashMap;
use log::*;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    StatusUpdate {
        status: String,
        timestamp: String,
    },
    ExecutionResult {
        output: String,
        metrics: HashMap<String, f64>,
    },
    Error {
        message: String,
        code: Option<i32>,
    },
    // Add more message types as needed
}

#[derive(Debug, Clone)]
pub struct MessageHandler {
    callbacks: HashMap<String, Vec<Callback<ServerMessage>>>,
}

impl Default for MessageHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageHandler {
    pub fn new() -> Self {
        Self {
            callbacks: HashMap::new(),
        }
    }

    pub fn register_handler(&mut self, message_type: &str, callback: Callback<ServerMessage>) {
        self.callbacks
            .entry(message_type.to_string())
            .or_default()
            .push(callback);
    }

    pub fn handle_message(&self, message: &str) -> Result<(), JsValue> {
        match serde_json::from_str::<ServerMessage>(message) {
            Ok(msg) => {
                let message_type = match &msg {
                    ServerMessage::StatusUpdate { .. } => "status_update",
                    ServerMessage::ExecutionResult { .. } => "execution_result",
                    ServerMessage::Error { .. } => "error",
                };

                if let Some(callbacks) = self.callbacks.get(message_type) {
                    for callback in callbacks {
                        callback.emit(msg.clone());
                    }
                }
                
                // Also send to any wildcard handlers
                if let Some(callbacks) = self.callbacks.get("*") {
                    for callback in callbacks {
                        callback.emit(msg.clone());
                    }
                }
                
                Ok(())
            }
            Err(e) => {
                error!("Failed to parse message: {:?}", e);
                Err(JsValue::from_str(&format!("Failed to parse message: {}", e)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    use yew::callback::Callback;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_message_handler() {
        let mut handler = MessageHandler::new();
        
        let status_received = std::rc::Rc::new(std::cell::Cell::new(false));
        let status_received_clone = status_received.clone();
        
        handler.register_handler(
            "status_update",
            Callback::from(move |_| {
                status_received_clone.set(true);
            }),
        );
        
        let message = r#"
            {
                "type": "status_update",
                "status": "processing",
                "timestamp": "2023-07-31T23:59:59Z"
            }
        "#;
        
        handler.handle_message(message).unwrap();
        assert!(status_received.get(), "Status update handler was not called");
    }
}
