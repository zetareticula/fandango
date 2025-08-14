use std::path::{Path, PathBuf};
use std::fs::{create_dir_all, File};
use std::io::Write;
use candle_core::{DType, Device, Tensor};
use rand::Rng;

/// Creates a test directory for model files if it doesn't exist
pub fn ensure_test_dir() -> PathBuf {
    let test_dir = Path::new("test/models");
    if !test_dir.exists() {
        create_dir_all(test_dir).expect("Failed to create test directory");
    }
    test_dir.to_path_buf()
}

/// Generates a small test model file with random weights
pub fn generate_test_model(path: &Path, rows: usize, cols: usize) -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut rng = rand::thread_rng();
    
    // Generate random weights
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Save to file in a simple binary format
    let mut file = File::create(path)?;
    file.write_all(&(rows as u32).to_le_bytes())?;
    file.write_all(&(cols as u32).to_le_bytes())?;
    
    // Convert to bytes and write
    let bytes: Vec<u8> = data
        .into_iter()
        .flat_map(|f| f.to_le_bytes().to_vec())
        .collect();
    file.write_all(&bytes)?;
    
    Ok(())
}

/// Creates a test model and returns its path
pub fn setup_test_model(name: &str, rows: usize, cols: usize) -> PathBuf {
    let test_dir = ensure_test_dir();
    let model_path = test_dir.join(format!("{}.bin", name));
    
    if !model_path.exists() {
        generate_test_model(&model_path, rows, cols)
            .expect("Failed to generate test model");
    }
    
    model_path
}

/// Helper function to create a test tensor
pub fn create_test_tensor(shape: &[usize]) -> Tensor {
    let device = Device::Cpu;
    let total: usize = shape.iter().product();
    let data: Vec<f32> = (0..total).map(|x| x as f32 / total as f32).collect();
    Tensor::from_slice(&data, shape, &device).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    
    #[test]
    fn test_generate_test_model() {
        let test_dir = ensure_test_dir();
        let model_path = test_dir.join("test_generate.bin");
        
        // Clean up if exists
        let _ = fs::remove_file(&model_path);
        
        // Generate test model
        generate_test_model(&model_path, 10, 10).unwrap();
        assert!(model_path.exists());
        
        // Verify file size
        let metadata = fs::metadata(&model_path).unwrap();
        assert_eq!(metadata.len(), 8 + 10 * 10 * 4); // 8 bytes for dimensions + 10x10 f32s
    }
}
