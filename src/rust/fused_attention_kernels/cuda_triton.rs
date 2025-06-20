#[cfg(feature = "cuda")]
use triton_client::inference::InferenceClient;
use candle_core::{Tensor, Device, Result, DType};
use half::f16;


pub struct TritonAttention {
    client: InferenceClient,
    device: Device,
}

impl TritonAttention {
    pub fn new(device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let client = InferenceClient::new("localhost:8001", "fandango-triton")?; // Triton server
        Ok(TritonAttention { client, device })
    }

    pub fn quantize_fp8(&self, input: &Tensor) -> Result<Tensor> {
        let input_data = input.to_vec1::<f32>()?;
        let fp8_data = input_data.iter().map(|&x| {
            let fp8 = f16::from_f32(x); // FP8 approximation
            let bits = fp8.to_bits() as u16;
            (bits as f32) / (u16::MAX as f32) * 255.0 // Scale to FP8 range
        }).collect::<Vec<f32>>();

        let output = self.client.infer(&fp8_data, "fp8_quantize")?; // Triton FP8 kernel
        Tensor::from_vec(output, input.shape(), &self.device)?.to_dtype(DType::F32)
    }
}