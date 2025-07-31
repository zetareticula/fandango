use candle_core::{Device, Tensor};
use fandango::fused_attention_kernels::FusedAttention;
use anyhow::Result;

fn main() -> Result<()> {
    println!("Testing FusedAttention implementation...");
    
    // Initialize the device (use CPU for compatibility)
    let device = Device::Cpu;
    println!("Using device: {:?}", device);
    
    // Initialize the FusedAttention module
    let input_dim = 64;
    let hidden_dim = 64;
    let num_heads = 8;
    let dropout = 0.1;
    
    println!("Initializing FusedAttention with input_dim={}, hidden_dim={}, num_heads={}, dropout={}",
             input_dim, hidden_dim, num_heads, dropout);
    
    let mut attention = FusedAttention::new(
        device.clone(),
        input_dim,
        hidden_dim,
        num_heads,
        dropout,
        None // No pre-trained weights
    )?;
    
    // Create a test input tensor
    let batch_size = 2;
    let seq_len = 10;
    let input_shape = (batch_size, seq_len, input_dim);
    
    println!("\nCreating test input with shape: {:?}", input_shape);
    
    // Create a simple input tensor with values from 0 to batch_size*seq_len*input_dim
    let total_elements = batch_size * seq_len * input_dim;
    let input_data: Vec<f32> = (0..total_elements).map(|x| x as f32 * 0.01).collect();
    
    // Process the input through the attention module
    println!("Processing input through attention module...");
    let output = attention.process_mermaid_flow(input_data)?;
    
    // Print some information about the output
    println!("\nAttention processing complete!");
    println!("Output length: {}", output.len());
    println!("First 5 output values: {:?}", &output[..5].to_vec());
    
    Ok(())
}
