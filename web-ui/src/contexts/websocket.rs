use std::cell::RefCell;
use std::rc::Rc;
use yew::prelude::*;
use yew::suspense::use_future;
use wasm_bindgen_futures::spawn_local;
use gloo_net::websocket::{futures::WebSocket, Message as WsMessage};
use futures_util::{SinkExt, StreamExt};
use log::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq)]
pub enum WebSocketStatus {
    Connected,
    Connecting,
    Disconnected,
    Error(String),
}

impl Default for WebSocketStatus {
    fn default() -> Self {
        WebSocketStatus::Disconnected
    }
}

#[derive(Debug, Clone)]
pub struct WebSocketMessage {
    pub data: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Default)]
pub struct WebSocketContextInner {
    pub status: WebSocketStatus,
    pub messages: VecDeque<WebSocketMessage>,
    pub error: Option<String>,
    pub ws: Option<WebSocket>,
}

#[derive(Debug, Clone)]
pub struct WebSocketContext {
    inner: Rc<RefCell<WebSocketContextInner>>,
}

impl PartialEq for WebSocketContext {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

#[derive(Properties, Debug, PartialEq)]
pub struct WebSocketProviderProps {
    #[prop_or_default]
    pub children: Children,
    pub url: String,
}

impl WebSocketContext {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(WebSocketContextInner::default())),
        }
    }

    pub fn connect(&self, url: &str) {
        let inner = self.inner.clone();
        let url = url.to_string();

        spawn_local(async move {
            // Update status to connecting
            {
                let mut inner = inner.borrow_mut();
                inner.status = WebSocketStatus::Connecting;
                inner.error = None;
            }

            match WebSocket::open(&url) {
                Ok((ws, _)) => {
                    let (mut write, mut read) = ws.split();
                    
                    // Update status to connected
                    {
                        let mut inner = inner.borrow_mut();
                        inner.status = WebSocketStatus::Connected;
                        inner.ws = Some(write);
                    }

                    // Start reading messages
                    while let Some(result) = read.next().await {
                        match result {
                            Ok(message) => {
                                if let WsMessage::Text(text) = message {
                                    let mut inner = inner.borrow_mut();
                                    inner.messages.push_back(WebSocketMessage {
                                        data: text,
                                        timestamp: chrono::Utc::now(),
                                    });
                                    
                                    // Keep only the last 100 messages
                                    if inner.messages.len() > 100 {
                                        inner.messages.pop_front();
                                    }
                                }
                            }
                            Err(e) => {
                                error!("WebSocket error: {:?}", e);
                                let mut inner = inner.borrow_mut();
                                inner.status = WebSocketStatus::Error(format!("WebSocket error: {:?}", e));
                                inner.error = Some(format!("WebSocket error: {:?}", e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to connect to WebSocket: {:?}", e);
                    let mut inner = inner.borrow_mut();
                    inner.status = WebSocketStatus::Error(format!("Failed to connect: {:?}", e));
                    inner.error = Some(format!("Failed to connect: {:?}", e));
                }
            }
        });
    }

    pub fn send(&self, message: &str) -> Result<(), String> {
        let inner = self.inner.borrow();
        if let Some(write) = &inner.ws {
            let mut write = write.clone();
            let message = message.to_string();
            
            spawn_local(async move {
                if let Err(e) = write.send(WsMessage::Text(message)).await {
                    error!("Failed to send WebSocket message: {:?}", e);
                }
            });
            
            Ok(())
        } else {
            Err("WebSocket not connected".to_string())
        }
    }

    pub fn status(&self) -> WebSocketStatus {
        self.inner.borrow().status.clone()
    }

    pub fn error(&self) -> Option<String> {
        self.inner.borrow().error.clone()
    }

    pub fn messages(&self) -> Vec<WebSocketMessage> {
        self.inner.borrow().messages.iter().cloned().collect()
    }
}

#[function_component(WebSocketProvider)]
pub fn websocket_provider(props: &WebSocketProviderProps) -> Html {
    let context = use_state(WebSocketContext::new);
    
    // Connect to WebSocket when component mounts
    {
        let context = context.clone();
        let url = props.url.clone();
        
        use_effect_with_deps(
            move |_| {
                context.connect(&url);
                || {}
            },
            (), // Only run once on mount
        );
    }
    
    html! {
        <ContextProvider<WebSocketContext> context={(*context).clone()}>
            {props.children.clone()}
        </ContextProvider<WebSocketContext>>
    }
}

pub fn use_websocket() -> WebSocketContext {
    use_context::<WebSocketContext>().expect("WebSocket context not found")
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    wasm_bindgen_test_configure!(run_in_browser);
    
    #[wasm_bindgen_test]
    fn test_websocket_context() {
        let context = WebSocketContext::new();
        assert_eq!(context.status(), WebSocketStatus::Disconnected);
        
        // Note: Actual WebSocket connection tests would require a WebSocket server
        // and are better suited for integration tests
    }
}
