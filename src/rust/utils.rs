//! Utility functions and helpers for the Fandango project.

use candle_core::Tensor;

/// A simple utility function to check if a tensor is empty
pub fn is_tensor_empty(tensor: &Tensor) -> bool {
    tensor.dims().iter().product::<usize>() == 0
}

/// A utility function to safely convert a tensor to a specific device
pub fn to_device(tensor: Tensor, device: &candle_core::Device) -> candle_core::Result<Tensor> {
    tensor.to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_is_tensor_empty() {
        let device = Device::Cpu;
        let empty_tensor = Tensor::zeros((0,), candle_core::DType::F32, &device).unwrap();
        assert!(is_tensor_empty(&empty_tensor));

        let non_empty_tensor = Tensor::zeros((1,), candle_core::DType::F32, &device).unwrap();
        assert!(!is_tensor_empty(&non_empty_tensor));
    }
}
