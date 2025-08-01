mod app;
mod components;
mod contexts;
mod pages;
mod services;
mod utils;

// Re-exports
pub use contexts::{WebSocketContext, WebSocketProvider, WebSocketStatus};

use wasm_bindgen::prelude::*;
use yew::prelude::*;
use yew::Renderer;
use web_sys::console;

use app::App;
use contexts::WebSocketProvider;

#[wasm_bindgen]
pub fn run_app() -> Result<(), JsValue> {
    // Initialize logging
    wasm_logger::init(wasm_logger::Config::default());
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    log::info!("Starting Fandango Web UI");
    
    // Get the current URL
    let window = web_sys::window().expect("no global `window` exists");
    let location = window.location();
    let protocol = location.protocol().unwrap_or_else(|_| "http:".to_string());
    let host = location.host().unwrap_or_else(|_| "localhost:8080".to_string());
    
    // Determine WebSocket URL based on current protocol
    let ws_protocol = if protocol == "https:" { "wss" } else { "ws" };
    let ws_url = format!("{}://{}/ws", ws_protocol, host);
    
    console::log_1(&format!("Connecting to WebSocket at: {}", ws_url).into());
    
    // Mount the app with WebSocket provider
    let app = html! {
        <WebSocketProvider url={ws_url}>
            <App />
        </WebSocketProvider>
    };
    
    Renderer::<App>::with_root_and_props(
        gloo_utils::document()
            .get_element_by_id("app")
            .expect("a #app element")
            .into(),
        app,
    )
    .render();
    
    Ok(())
}
