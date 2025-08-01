use yew::prelude::*;
use yew::context::ContextHandle;
use std::rc::Rc;
use log::*;
use wasm_bindgen_futures::spawn_local;

use crate::services::websocket::{WebSocketService, WsMessage};

#[derive(Debug)]
pub struct WebSocketContextInner {
    pub service: WebSocketService,
    pub status: WebSocketStatus,
    pub last_error: Option<String>,
}

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

#[derive(Debug, Default)]
pub struct WebSocketContext {
    inner: Rc<WebSocketContextInner>,
    _listener: ContextHandle<WebSocketContext>,
}

impl WebSocketContext {
    pub fn new(service: WebSocketService) -> Self {
        let inner = Rc::new(WebSocketContextInner {
            service,
            status: WebSocketStatus::Disconnected,
            last_error: None,
        });
        
        let inner_clone = inner.clone();
        let _listener = ContextHandle::new(Box::new(move |_| {
            // Reconnect logic can be implemented here
        }));
        
        Self { inner, _listener }
    }
    
    pub fn connect(&self) {
        let inner = self.inner.clone();
        
        // Update status to connecting
        inner.status = WebSocketStatus::Connecting;
        
        // Set up callbacks
        let on_open = {
            let inner = inner.clone();
            Callback::from(move |_| {
                inner.status = WebSocketStatus::Connected;
                inner.last_error = None;
                info!("WebSocket connected");
            })
        };
        
        let on_close = {
            let inner = inner.clone();
            Callback::from(move |_| {
                inner.status = WebSocketStatus::Disconnected;
                info!("WebSocket disconnected");
            })
        };
        
        let on_error = {
            let inner = inner.clone();
            Callback::from(move |error: String| {
                inner.status = WebSocketStatus::Error(error.clone());
                inner.last_error = Some(error);
                error!("WebSocket error: {:?}", inner.last_error);
            })
        };
        
        // Set up the WebSocket service
        inner.service.on_open = Some(on_open);
        inner.service.on_close = Some(on_close);
        inner.service.on_error = Some(on_error);
        
        // Connect
        if let Err(e) = inner.service.connect() {
            inner.status = WebSocketStatus::Error(e.clone());
            inner.last_error = Some(e);
        }
    }
    
    pub fn disconnect(&self) {
        self.inner.service.disconnect();
    }
    
    pub fn send(&self, message: &str) -> Result<(), String> {
        self.inner.service.send(message)
    }
    
    pub fn status(&self) -> WebSocketStatus {
        self.inner.status.clone()
    }
    
    pub fn last_error(&self) -> Option<String> {
        self.inner.last_error.clone()
    }
}

// Context provider component
#[derive(Properties, Debug, PartialEq)]
pub struct WebSocketProviderProps {
    #[prop_or_default]
    pub children: Children,
    pub url: String,
}

#[function_component(WebSocketProvider)]
pub fn websocket_provider(props: &WebSocketProviderProps) -> Html {
    let websocket_service = use_memo(
        |url| WebSocketService::new(url.clone()),
        props.url.clone(),
    );
    
    let context = use_memo(
        |service| WebSocketContext::new(service.clone()),
        (*websocket_service).clone(),
    );
    
    // Connect when component mounts
    use_effect_with_deps(
        |ctx| {
            ctx.connect();
            || ctx.disconnect()
        },
        context.clone(),
    );
    
    html! {
        <ContextProvider<WebSocketContext> context={(*context).clone()}>
            {props.children.clone()}
        </ContextProvider<WebSocketContext>>
    }
}

// Hook to use the WebSocket context
pub fn use_websocket() -> WebSocketContext {
    use_context::<WebSocketContext>().expect("WebSocket context not found")
}
