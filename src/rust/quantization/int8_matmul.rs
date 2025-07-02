use candle_core::{Tensor, Device, Result, DType};
use std::ops::Mul;

pub fn int8_matmul(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    // Convert to Int8, assuming inputs are pre-normalized
    let a_int8 = a.to_dtype(DType::I8)?;
    let b_int8 = b.to_dtype(DType::I8)?;

    // Perform Int8 matrix multiplication
    let result = a_int8.matmul(&b_int8)?;

    // Dequantize back to FP32 for further processing
    let scale = 1.0 / 127.0; // Simple scaling factor (adjust based on calibration)
    result.to_dtype(DType::F32)?.mul(scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_matmul() -> Result<()> {
        let device = Device::Cuda(0)?;
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((2, 2))?;
        let b = Tensor::new(&[5.0f32, 6.0, 7.0, 8.0], &device)?.reshape((2, 2))?;
        let result = int8_matmul(&a, &b, &device)?;
        assert_eq!(result.dim(0)?, 2);
        assert_eq!(result.dim(1)?, 2);
        Ok(())
    }
}