use candle_core::Device;
use fandango::fused_attention_kernels::FusedAttention;
use anyhow::{Result, Context};
use log::{info, error};
use simple_logger::SimpleLogger;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .context("Failed to initialize logger")?;

    info!("Starting Fandango application...");
    
    // Initialize the device (use CPU for now, can be changed to CUDA if available)
    let device = Device::Cpu;
    info!("Using device: {:?}", device);
    
    // Initialize the FusedAttention
    info!("Initializing FusedAttention...");
    let mut attention = FusedAttention::new(
        device.clone(),
        64,  // input_dim
        64,  // hidden_dim
        8,   // num_heads
        0.1, // dropout
        None // Optional: path to pre-trained weights
    ).context("Failed to initialize FusedAttention")?;
    
    // Create a sample input tensor
    let input = vec![1.0; 64];
    info!("Created input tensor with {} elements", input.len());
    
    // Process the input
    info!("Processing input...");
    match attention.process_mermaid_flow(input).await {
        Ok(output) => {
            info!("Successfully processed input");
            info!("Output length: {}", output.len());
            info!("First 5 output values: {:?}", &output[..5]);
        }
        Err(e) => {
            error!("Error processing input: {}", e);
            return Err(e.into());
        }
    }
    
    info!("Application completed successfully");
    Ok(())
}