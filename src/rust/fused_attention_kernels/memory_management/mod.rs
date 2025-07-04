//! Memory management utilities for the fused attention kernels.

use candle_core::{Tensor, Device, DType};
use candle_core::error::Result as CandleResult;
use crate::fused_attention_kernels::distiller::monitoring;

/// Manages memory allocation and deallocation for the attention mechanism.
pub struct MemoryManager {
    device: Device,
    dram_percentage: f32, // % of model in DRAM (0-1)
    opt_67b_params: Tensor, // OPT-6.7B parameters placeholder
}

impl MemoryManager {
    /// Creates a new MemoryManager instance.
    pub fn new(device: Device) -> CandleResult<Self> {
        // Start with a smaller tensor for testing
        let param_size = 1_000_000; // Reduced size for testing
        let opt_67b_params = Tensor::zeros((param_size,), DType::F32, &device)?; // Placeholder
        
        Ok(MemoryManager {
            device,
            dram_percentage: 0.5, // Default 50% in DRAM
            opt_67b_params,
        })
    }

    /// Adjusts the DRAM usage percentage and returns the parameters in DRAM.
    pub fn adjust_dram_usage(&mut self, percentage: f32) -> CandleResult<Tensor> {
        self.dram_percentage = percentage.clamp(0.0, 1.0);
        let dram_size = (self.opt_67b_params.num_elements()? as f32 * self.dram_percentage) as usize;
        let dram_params = self.opt_67b_params.narrow(0, 0, dram_size)?;
        let start = std::time::Instant::now();
        let output = dram_params.clone(); // Simulate usage
        let latency = start.elapsed().as_secs_f64() * 1000.0; // ms
        monitoring::record_latency(latency);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager() -> CandleResult<()> {
        let device = Device::Cpu;
        let mut manager = MemoryManager::new(device)?;
        
        // Test default DRAM percentage
        assert_eq!(manager.dram_percentage, 0.5);
        
        // Test adjusting DRAM usage
        let params = manager.adjust_dram_usage(0.75)?;
        assert_eq!(manager.dram_percentage, 0.75);
        assert!(params.dims() == &[750_000]); // 75% of 1,000,000
        
        // Test bounds checking
        let _ = manager.adjust_dram_usage(1.5)?;
        assert_eq!(manager.dram_percentage, 1.0);
        
        let _ = manager.adjust_dram_usage(-0.5)?;
        assert_eq!(manager.dram_percentage, 0.0);
        
        Ok(())
    }
}
