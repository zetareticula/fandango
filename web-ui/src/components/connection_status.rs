use yew::prelude::*;
use yew_feather::{Wifi, WifiOff, RefreshCw, AlertCircle};
use stylist::{style, yew::styled_component};
use crate::contexts::use_websocket;
use crate::contexts::WebSocketStatus;

#[derive(Properties, PartialEq)]
pub struct ConnectionStatusProps {
    pub server_url: String,
}

#[styled_component(ConnectionStatus)]
pub fn connection_status(props: &ConnectionStatusProps) -> Html {
    let ws = use_websocket();
    let status = ws.status();
    let error = ws.error();
    
    let (status_text, status_color, icon) = match status {
        WebSocketStatus::Connected => ("Connected", "#10b981", html! { <Wifi size={16} color="#10b981" /> }),
        WebSocketStatus::Connecting => ("Connecting...", "#f59e0b", html! { <RefreshCw size={16} class="animate-spin" color="#f59e0b" /> }),
        WebSocketStatus::Disconnected => ("Disconnected", "#ef4444", html! { <WifiOff size={16} color="#ef4444" /> }),
        WebSocketStatus::Error(_) => ("Connection Error", "#ef4444", html! { <AlertCircle size={16} color="#ef4444" /> }),
    };
    
    let status_style = style!(
        r#"
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        background-color: var(--color-bg-secondary);
        
        .status-dot {
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 50%;
            background-color: ${status_color};
        }
        
        .server-url {
            opacity: 0.7;
            font-family: monospace;
            font-size: 0.75rem;
        }
        
        .error-message {
            color: #ef4444;
            font-size: 0.75rem;
            margin-left: 0.5rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 200px;
        }
        
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        "#,
        status_color = status_color
    )
    .unwrap();
    
    let reconnect = {
        let ws = ws.clone();
        let url = props.server_url.clone();
        Callback::from(move |_| {
            ws.connect(&url);
        })
    };

    html! {
        <div class={status_style}>
            {icon}
            <span class="status-dot"></span>
            <span>{status_text}</span>
            <span class="server-url">{&props.server_url}</span>
            
            if let WebSocketStatus::Error(_) = status {
                <button 
                    class="ml-2 px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                    onclick={reconnect}
                >
                    {"Reconnect"}
                </button>
            }
            
            if let Some(err) = error {
                <span class="error-message" title={err.clone()}>
                    {err}
                </span>
            }
        </div>
    }
}
