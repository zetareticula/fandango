//! Application contexts for global state management

pub mod websocket;

// Re-export for convenience
pub use websocket::{WebSocketContext, WebSocketProvider, WebSocketStatus};
