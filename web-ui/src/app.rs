use yew::prelude::*;
use yew_router::prelude::*;
use yew_feather::{Activity, Settings, Terminal};

use crate::{
    components::connection_status::ConnectionStatus,
    pages::{home::HomePage, not_found::NotFoundPage},
};

/// Main application component
#[function_component(App)]
pub fn app() -> Html {
    // Get the current host for WebSocket connection
    let window = web_sys::window().expect("no global `window` exists");
    let location = window.location();
    let host = location.host().unwrap_or_else(|_| "localhost:8080".to_string());
    let ws_url = format!("ws://{}/ws", host);
    
    html! {
        <BrowserRouter>
            <div class="app-container">
                <header class="app-header">
                    <div class="container flex items-center justify-between py-4">
                        <div class="flex items-center gap-4">
                            <Terminal size={24} class="text-primary" />
                            <h1 class="text-xl font-bold m-0">{"Fandango"}</h1>
                        </div>
                        <nav class="flex items-center gap-4">
                            <Link<Route> to={Route::Home} classes="px-3 py-2 rounded hover:bg-gray-100">
                                {"Home"}
                            </Link<Route>>
                            <Link<Route> to={Route::Home} classes="px-3 py-2 rounded hover:bg-gray-100">
                                <Settings size={20} />
                            </Link<Route>>
                        </nav>
                    </div>
                    <div class="connection-bar">
                        <div class="container">
                            <ConnectionStatus server_url={ws_url.clone()} />
                        </div>
                    </div>
                </header>
                
                <main class="main-content">
                    <div class="container">
                        <Switch<Route> render={switch} />
                    </div>
                </main>
                
                <footer class="py-4 border-t border-gray-200 mt-auto">
                    <div class="container text-center text-sm text-gray-500">
                        {"© 2023 Fandango - High-performance attention mechanisms for deep learning"}
                    </div>
                </footer>
            </div>
        </BrowserRouter>
    }
}

/// Application routes
#[derive(Clone, Routable, PartialEq, Debug)]
pub enum Route {
    #[at("/")]
    Home,
    #[not_found]
    #[at("/404")]
    NotFound,
}

/// Route switcher
fn switch(route: Route) -> Html {
    match route {
        Route::Home => html! { <HomePage /> },
        Route::NotFound => html! { <NotFoundPage /> },
    }
}
