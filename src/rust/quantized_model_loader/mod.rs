use crate::kvcache_manager::monitoring::Monitoring;
use crate::kvcache_manager::deduplication::deduplicate_kvcache;
use crate::kvcache_manager::precision::{Precision, calculate_entropy, measure_locality};
use std::sync::Arc;
use tokio::sync::Mutex;
use candle_core::Result;
use candle_core::Device;
use candle_core::CudaDevice;

use crate::kvcache_manager::deduplication::deduplicate_kvcache;
use crate::kvcache_manager::precision::{Precision, calculate_entropy, measure_locality};


#[derive(Debug, Clone)]
pub struct quantized_model_loader {
    pub device: Device,
    pub load_level: f32,
    pub current_precision: Precision,
}

#[derive(Debug, Clone)]


if cfg(feature = "cuda") {
    use candle_core::CudaDevice;
}


#[derive(Debug, Clone)]
pub fn init_kvcache_manager(device: Device) -> KVCacheManager {
    KVCacheManager {
        device,
        load_level: 0.0,
        current_precision: Precision::Int8, // Default precision
    }
}