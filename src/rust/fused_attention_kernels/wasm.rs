#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use candle_core::{Device, Tensor, DType};
use super::FusedAttention;


/// Initialize a new FusedAttention instance for WASM
#[wasm_bindgen]
pub fn init_wasm_attention() -> *mut FusedAttention {
    let device = Device::Cpu; // WASM runs on CPU
    // Initialize with default parameters
    let attention = match FusedAttention::new(
        device,
        512,  // input_dim
        512,  // hidden_dim
        8,    // num_heads
        0.1,  // dropout
        None, // weights_path
    ) {
        Ok(attn) => attn,
        Err(e) => {
            web_sys::console::error_1(&format!("Failed to initialize attention: {}", e).into());
            panic!("Failed to initialize attention");
        }
    };
    Box::into_raw(Box::new(attention))
}

/// Apply attention to the input query
#[wasm_bindgen]
pub fn apply_wasm_attention(ptr: *mut FusedAttention, query: Vec<f32>) -> Vec<f32> {
    let attention = unsafe { &mut *ptr };
    let device = Device::Cpu;
    
    // Convert input to tensor
    let tensor = match Tensor::from_vec(query.clone(), (query.len(), 1), &device) {
        Ok(t) => t,
        Err(e) => {
            web_sys::console::error_1(&format!("Tensor creation failed: {}", e).into());
            return vec![];
        }
    };
    
    // Apply attention
    match attention.forward(&tensor, None, None, false) {
        Ok(output) => match output.to_vec1::<f32>() {
            Ok(vec) => vec,
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to convert output: {}", e).into());
                vec![]
            }
        },
        Err(e) => {
            web_sys::console::error_1(&format!("Attention forward failed: {}", e).into());
            vec![]
        }
    }
}

/// Free the FusedAttention instance
#[wasm_bindgen]
pub fn free_wasm_attention(ptr: *mut FusedAttention) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr); // This will drop the FusedAttention instance
        }
    }
}

