#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cuda(0)?;
    let mut attention = FusedAttention::new(device)?;
    let input = vec![1.0; 64];
    let output = attention.process_mermaid_flow(input)?;
    println!("Output: {:?}", output);
    std::thread::sleep(std::time::Duration::from_secs(10)); // Simulate load
    Ok(())
}


#![cfg(feature = "cuda")]
use candle_core::CudaDevice;
use candle_core::{Result, Device};
use fused_attention_kernels::FusedAttention;
use fused_attention_kernels::cuda_triton::TritonAttention;
use kvcache_manager::KVCacheManager;
use runtime_scheduler::TokenWindow;
use runtime_scheduler::RuntimeScheduler;

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn init_wasm_attention() -> *mut FusedAttention {
    let device = Device::Cuda(0).unwrap(); // Use CUDA device
    let attention = FusedAttention::new(device).unwrap();
    Box::into_raw(Box::new(attention))
}


#[wasm_bindgen::prelude::wasm_bindgen]
pub fn apply_wasm_attention(ptr: *mut FusedAttention, query: Vec<f32>) -> Vec<f32> {
    let attention = unsafe { &mut *ptr };
    let tensor = candle_core::Tensor::from_vec(query, (query.len(), 1), &attention.device).unwrap();
    let output = attention.process_mermaid_flow(tensor.to_vec1::<f32>().unwrap()).unwrap();
    output
}

#[wasm_bindgen::prelude::wasm_bindgen]
pub fn free_wasm_attention(ptr: *mut FusedAttention) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr); // This will drop the FusedAttention instance
    }
}

use candle_core::Result;
use candle_core::Device;
use fused_attention_kernels::FusedAttention;
use fused_attention_kernels::cuda_triton::TritonAttention;
use kvcache_manager::KVCacheManager;
use runtime_scheduler::TokenWindow;
use runtime_scheduler::RuntimeScheduler;

#[cfg(feature = "cuda")]
use candle_core::CudaDevice;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub fn init_wasm_attention() -> *mut FusedAttention {
    let device = Device::Cuda(0).unwrap(); // Use CUDA device
    let attention = FusedAttention::new(device).unwrap();
    Box::into_raw(Box::new(attention))
}