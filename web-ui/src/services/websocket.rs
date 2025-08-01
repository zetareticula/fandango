use gloo_net::websocket::{futures::WebSocket, Message as WsMessage};
use wasm_bindgen_futures::spawn_local;
use yew::callback::Callback;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::cell::RefCell;
use log::*;

pub type WsCallback = Callback<Result<WsMessage, gloo_net::websocket::WebSocketError>>;

#[derive(Debug, Clone)]
pub struct WebSocketService {
    pub url: String,
    pub ws: Option<WebSocket>,
    pub on_message: Option<WsCallback>,
    pub on_open: Option<Callback<()>>,
    pub on_close: Option<Callback<()>>,
    pub on_error: Option<Callback<String>>,
}

impl WebSocketService {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            ws: None,
            on_message: None,
            on_open: None,
            on_close: None,
            on_error: None,
        }
    }

    pub fn connect(&mut self) -> Result<(), String> {
        // Close existing connection if any
        self.disconnect();

        match WebSocket::open(&self.url) {
            Ok(ws) => {
                let ws_rc = Rc::new(RefCell::new(Some(ws)));
                let ws_clone = ws_rc.clone();
                
                // Clone callbacks
                let on_open = self.on_open.clone();
                let on_close = self.on_close.clone();
                let on_error = self.on_error.clone();
                let on_message = self.on_message.clone();

                // Spawn a future to handle WebSocket messages
                spawn_local(async move {
                    let mut ws_opt = ws_clone.borrow_mut();
                    if let Some(mut ws) = ws_opt.take() {
                        // Notify connection opened
                        if let Some(cb) = on_open {
                            cb.emit(());
                        }

                        while let Some(result) = ws.next().await {
                            match result {
                                Ok(message) => {
                                    if let Some(cb) = on_message.as_ref() {
                                        cb.emit(Ok(message));
                                    }
                                }
                                Err(e) => {
                                    error!("WebSocket error: {:?}", e);
                                    if let Some(cb) = on_error.as_ref() {
                                        cb.emit(format!("WebSocket error: {:?}", e));
                                    }
                                    break;
                                }
                            }
                        }

                        // Notify connection closed
                        if let Some(cb) = on_close {
                            cb.emit(());
                        }
                    }
                });

                self.ws = ws_rc.borrow_mut().take();
                Ok(())
            }
            Err(e) => {
                error!("Failed to connect to WebSocket: {:?}", e);
                if let Some(cb) = self.on_error.as_ref() {
                    cb.emit(format!("Failed to connect: {:?}", e));
                }
                Err("Failed to connect to WebSocket".to_string())
            }
        }
    }

    pub fn disconnect(&mut self) {
        if let Some(ws) = self.ws.take() {
            // Close the WebSocket connection
            // Note: The WebSocket will be dropped when this function returns
            if let Some(cb) = self.on_close.as_ref() {
                cb.emit(());
            }
        }
    }

    pub fn send(&mut self, message: &str) -> Result<(), String> {
        if let Some(ws) = &mut self.ws {
            match ws.send(WsMessage::Text(message.to_string())) {
                Ok(_) => Ok(()),
                Err(e) => {
                    error!("Failed to send WebSocket message: {:?}", e);
                    if let Some(cb) = self.on_error.as_ref() {
                        cb.emit(format!("Failed to send message: {:?}", e));
                    }
                    Err("Failed to send message".to_string())
                }
            }
        } else {
            Err("WebSocket not connected".to_string())
        }
    }

    pub fn is_connected(&self) -> bool {
        self.ws.is_some()
    }
}

impl Drop for WebSocketService {
    fn drop(&mut self) {
        self.disconnect();
    }
}
