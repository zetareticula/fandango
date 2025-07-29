use yew::prelude::*;
use yew_router::prelude::*;

use crate::app::Route;

/// 404 Not Found page
#[function_component(NotFoundPage)]
pub fn not_found_page() -> Html {
    html! {
        <div class="flex flex-col items-center justify-center h-screen bg-gray-100">
            <h1 class="text-4xl font-bold text-gray-800 mb-4">{"404 - Page Not Found"}</h1>
            <p class="text-gray-600 mb-6">
                {"The page you are looking for doesn't exist or has been moved."}
            </p>
            <Link<Route> to={Route::Home} classes="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                {"Go to Home"}
            </Link<Route>>
        </div>
    }
}
