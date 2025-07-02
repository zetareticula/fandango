use candle_core::Device;
use fandango::fused_attention_kernels::FusedAttention;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the device (use CPU for now, can be changed to CUDA if available)
    let device = Device::Cpu;
    
    // Initialize the FusedAttention
    let mut attention = FusedAttention::new(
        device.clone(),
        64,  // input_dim
        64,  // hidden_dim
        8,   // num_heads
        0.1, // dropout
        None // Optional: path to pre-trained weights
    )?;
    
    // Create a sample input tensor
    let input = vec![1.0; 64];
    
    // Process the input
    let output = attention.process_mermaid_flow(input).await?;
    
    println!("Output: {:?}", output);
    Ok(())
}