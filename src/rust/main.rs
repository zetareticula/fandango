#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cuda(0)?;
    let mut attention = FusedAttention::new(device)?;
    let input = vec![1.0; 64];
    let output = attention.process_mermaid_flow(input)?;
    println!("Output: {:?}", output);
    Ok(())
}