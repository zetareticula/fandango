use candle_core::{Tensor, DType, Device};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MemoryLayoutError {
    #[error("Memory layout is full")]
    OutOfMemory,
    #[error("Tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),
}

type Result<T> = std::result::Result<T, MemoryLayoutError>;

/// FFNMemoryLayout is a structure that holds the memory layout for a feed-forward network (FFN).
/// It manages the matrix of weights, pointers to original neuron indices, biases, and tracks the number of utilized rows and last k active neurons.
pub struct FFNMemoryLayout {
    matrix: Tensor,         // Concatenated up/down project rows
    pointers: Vec<usize>,   // Original neuron indices
    bias: Vec<f32>,         // Up project bias
    num_used: usize,        // Number of utilized rows
    last_k_active: Vec<usize>, // Last k active neurons
}

impl FFNMemoryLayout {
    pub fn new(layer_size: usize) -> Result<Self> {
        let device = Device::Cpu;
        let matrix = Tensor::zeros((layer_size, 128), DType::F32, &device)
            .map_err(MemoryLayoutError::from)?;
        
        Ok(FFNMemoryLayout {
            matrix,
            pointers: Vec::with_capacity(layer_size),
            bias: vec![0.0; layer_size],
            num_used: 0,
            last_k_active: Vec::new(),
        })
    }

    pub fn add_neuron(&mut self, neuron_idx: usize, weight_row: Vec<f32>, bias: f32) -> Result<()> {
        let total_rows = self.matrix.dim(0).map_err(MemoryLayoutError::from)?;
        if self.num_used >= total_rows {
            return Err(MemoryLayoutError::OutOfMemory);
        }
        
        // Create a new tensor for the weight row
        let weight_tensor = Tensor::from_vec(weight_row, (1, 128), self.matrix.device())
            .map_err(MemoryLayoutError::from)?;
        
        if self.num_used == 0 {
            // First row - just assign directly
            self.matrix = weight_tensor;
        } else {
            // For subsequent rows, we need to concatenate
            let top = self.matrix.narrow(0, 0, self.num_used)
                .map_err(MemoryLayoutError::from)?;
            self.matrix = Tensor::cat(&[&top, &weight_tensor], 0)
                .map_err(MemoryLayoutError::from)?;
        }
        
        self.pointers.push(neuron_idx);
        self.bias[self.num_used] = bias;
        self.num_used += 1;
        self.last_k_active.push(neuron_idx);
        
        Ok(())
    }
}