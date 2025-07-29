use yew::prelude::*;
use yew_router::prelude::*;

use crate::pages::{home::HomePage, not_found::NotFoundPage};

/// Main application component
#[function_component(App)]
pub fn app() -> Html {
    html! {
        <BrowserRouter>
            <Switch<Route> render={switch} />
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
