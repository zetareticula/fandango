mod app;
mod components;
mod pages;
mod services;
mod utils;

use wasm_bindgen::prelude::*;
use yew::prelude::*;
use yew::Renderer;

use app::App;

#[wasm_bindgen]
pub fn run_app() -> Result<(), JsValue> {
    // Initialize logging
    wasm_logger::init(wasm_logger::Config::default());
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    log::info!("Starting Fandango Web UI");
    
    // Mount the app
    Renderer::<App>::new().render();
    
    Ok(())
}
