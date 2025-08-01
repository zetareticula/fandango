
// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.


//! Error handling for the Fandango integration module

use thiserror::Error;
use candle_core::Error as CandleError;
use std::io::Error as IoError;

/// Main error type for the integration module
#[derive(Error, Debug)]
pub enum IntegrationError {
    /// Error from the candle library
    #[error("Candle error: {0}")]
    Candle(#[from] CandleError),
    
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] IoError),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    
    /// Custom error message
    #[error("{0}")]
    Message(String),
    
    /// Component-specific errors
    #[error("Component error: {0}")]
    Component(String),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    Config(String),
    
    /// Operation not supported
    #[error("Operation not supported: {0}")]
    Unsupported(String),
}

/// Result type for the integration module
pub type Result<T> = std::result::Result<T, IntegrationError>;

impl From<&str> for IntegrationError {
    fn from(s: &str) -> Self {
        IntegrationError::Message(s.to_string())
    }
}

impl From<String> for IntegrationError {
    fn from(s: String) -> Self {
        IntegrationError::Message(s)
    }
}

/// Extension trait for converting between error types
pub trait ResultExt<T> {
    /// Convert to an integration error with context
    fn with_context<C: std::fmt::Display>(self, context: C) -> Result<T>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> ResultExt<T> for std::result::Result<T, E> {
    fn with_context<C: std::fmt::Display>(self, context: C) -> Result<T> {
        self.map_err(|e| IntegrationError::Message(format!("{}: {}", context, e)))
    }
}
