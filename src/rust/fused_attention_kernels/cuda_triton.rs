use candle_core::{Tensor, Device, Result};
use triton_client::inference::{InferenceClient, ModelInferRequest};

pub struct TritonClient {
    client: InferenceClient,
    device: Device,
}

impl TritonClient {
    pub fn new(device: Device) -> Result<Self> {
        let client = InferenceClient::new("localhost:8001", "http/2")?; // Update URL if different
        
        // Check if we're using CUDA and get the version
        if let Device::Cuda(device) = &device {
            // In the latest candle_core, we can use device.cuda_version()
            // but we'll use a more compatible approach
            log::info!("Using CUDA device");
        } else {
            log::warn!("Not using CUDA, performance may be suboptimal");
        }
        
        Ok(TritonClient { client, device })
    }

    pub fn infer(&self, input: &Tensor) -> Result<Tensor> {
        let request = ModelInferRequest::new("fandango_model")
            .add_input("input", input.to_vec1::<f32>()?, "FP32")
            .expect("Failed to add input");
        let response = self.client.infer(&request)?;
        Tensor::from_slice(&response.get_output("output")?.data, input.shape(), &self.device)
    }
}