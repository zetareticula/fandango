use proptest::prelude::*;
use candle_core::{DType, Device, Tensor};
use quantization_server::QuantizedModel;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    #[test]
    fn test_quantization_roundtrip(
        size in 1..=256usize,
        min_val in -10f32..0f32,
        max_val in 0.1f32..10f32
    ) {
        let device = Device::Cpu;
        // Create a tensor with values in the specified range
        let tensor = Tensor::rand(min_val, max_val, &[size, size], &device).unwrap();
        
        // Quantize and dequantize
        let quantized = QuantizedModel::quantize(&tensor, 8).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        
        // Check that the dequantized values are close to the original
        let diff = (tensor - &dequantized).unwrap();
        let max_diff = diff.abs().max().unwrap().to_scalar::<f32>().unwrap();
        
        // Allow for quantization error
        let range = max_val - min_val;
        let max_allowed_error = range / 128.0; // 8-bit quantization
        
        assert!(
            max_diff <= max_allowed_error * 1.1, // 10% tolerance
            "Max difference {} exceeds allowed error {}",
            max_diff,
            max_allowed_error
        );
    }
    
    #[test]
    fn test_quantized_matmul_consistency(
        m in 1..=32usize,
        k in 1..=32usize,
        n in 1..=32usize
    ) {
        let device = Device::Cpu;
        
        // Create random input and weight tensors
        let input = Tensor::randn(0f32, 1.0, &[m, k], &device).unwrap();
        let weights = Tensor::randn(0f32, 1.0, &[k, n], &device).unwrap();
        
        // Compute reference result
        let reference = input.matmul(&weights).unwrap();
        
        // Quantize weights and compute matmul
        let quantized = QuantizedModel::quantize(&weights, 8).unwrap();
        let result = quantized.quantized_matmul(&input).unwrap();
        
        // Check that results are close
        let diff = (reference - &result).unwrap();
        let max_diff = diff.abs().max().unwrap().to_scalar::<f32>().unwrap();
        
        // Allow for quantization error
        assert!(
            max_diff < 0.1,
            "Max difference {} exceeds threshold",
            max_diff
        );
    }
}

// Test different quantization bit widths
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]
    
    #[test]
    fn test_different_bit_widths(
        bits in 2..=8usize,
        size in 4..=64usize
    ) {
        let device = Device::Cpu;
        let tensor = Tensor::randn(0f32, 1.0, &[size, size], &device).unwrap();
        
        let quantized = QuantizedModel::quantize(&tensor, bits).unwrap();
        let dequantized = quantized.dequantize().unwrap();
        
        let diff = (tensor - &dequantized).unwrap();
        let mse = (diff.sqr().unwrap().mean_all().unwrap())
            .to_scalar::<f32>().unwrap();
            
        // Higher bit width should have lower error
        let expected_max_mse = 1.0 / ((1 << bits) as f32).powi(2);
        assert!(
            mse <= expected_max_mse,
            "MSE {} exceeds expected maximum {}",
            mse,
            expected_max_mse
        );
    }
}
