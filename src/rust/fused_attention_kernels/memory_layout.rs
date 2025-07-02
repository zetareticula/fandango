use candle_core::{Tensor, Device, DType, Result};

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
        let matrix = Tensor::zeros((layer_size, 128), DType::F32, &device)?;
        
        Ok(FFNMemoryLayout {
            matrix,
            pointers: Vec::with_capacity(layer_size),
            bias: vec![0.0; layer_size],
            num_used: 0,
            last_k_active: Vec::new(),
        })
    }

    pub fn add_neuron(&mut self, neuron_idx: usize, weight_row: Vec<f32>, bias: f32) -> Result<()> {
        if self.num_used >= self.matrix.dim(0)? {
            return Err(candle_core::Error::Msg("Memory layout full".to_string()));
        }
        
        self.pointers.push(neuron_idx);
        self.bias[self.num_used] = bias;
        
        // Create a new tensor for the weight row
        let weight_tensor = Tensor::from_vec(weight_row, (1, 128), self.matrix.device())?;
        
        // Get the slice of the matrix we want to update
        let mut matrix_slice = self.matrix.narrow(0, self.num_used, 1)?;
        
        // Use assign to copy the data from weight_tensor to matrix_slice
        matrix_slice.assign(&weight_tensor)?;
        
        self.num_used += 1;
        self.last_k_active.push(neuron_idx);
        
        Ok(())
    }
}