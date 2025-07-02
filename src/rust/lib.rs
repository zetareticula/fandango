// lib.rs

//! Fandango is a simple library for demonstration purposes.
//! It provides basic functionality for managing models and scheduling tasks.
//! This file is part of the Zeta Reticula - Fandango project, which is licensed under the Apache License 2.0.

use candle_core::{Result, DType, Device, Tensor, Error};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FandangoError {
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Tensor error: {0}")]
    TensorError(String),
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    #[error(transparent)]
    CandleError(#[from] candle_core::Error),
}

pub mod fused_attention_kernels;
pub mod kvcache_manager;
pub mod runtime_scheduler;
pub mod utils;

#[cfg(feature = "quantized_model_loader")]
pub mod quantized_model_loader;

#[cfg(feature = "fused_attention")]
pub mod fused_attention;

#[cfg(feature = "cuda")]
pub mod cuda_triton;

#[cfg(feature = "wasm")]
pub mod wasm;

/// Initialize zeta-reticula 
/// This function sets up the necessary components for the library to function correctly.
pub fn init() -> Result<()> {
    // Initialize the device, tensors, and other components as needed
    let device = Device::Cpu; // Default to CPU, can be changed to CUDA or other devices
    let dtype = DType::F32; // Default data type, can be changed as needed

    // Create a zero tensor
    let tensor = Tensor::zeros((1, 1), dtype, &device)?;

    // Compare device types
    if !tensor.device().same_device(&device) {
        return Err(Error::Msg("Failed to initialize tensor on the specified device".to_string()));
    }
    Ok(())
}

/// This function is used to set the device for the library.
/// It allows the user to specify which device (CPU, CUDA, etc.) to use for computations.
/// # Arguments
/// * `device` - The device to set for the library.
pub fn set_device(device: Device) -> Result<()> {
    // Simple device validation
    match device {
        Device::Cpu => {
            // CPU is always supported
            Ok(())
        },
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            // For CUDA, we'll just check if we can create a simple tensor
            Tensor::zeros((1,), DType::F32, &device)?;
            Ok(())
        },
        _ => Err(Error::Msg("Unsupported device type".to_string())),
    }
}






