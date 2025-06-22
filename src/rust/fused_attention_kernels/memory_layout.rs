use candle_core::Tensor;

pub struct FFNMemoryLayout {
    matrix: Tensor,         // Concatenated up/down project rows
    pointers: Vec<usize>,   // Original neuron indices
    bias: Vec<f32>,         // Up project bias
    num_used: usize,        // Number of utilized rows
    last_k_active: Vec<usize>, // Last k active neurons
}

impl FFNMemoryLayout {
    pub fn new(layer_size: usize) -> Result<Self, candle_core::Error> {
        let matrix = Tensor::zeros((layer_size, 128), candle_core::DType::F32, candle_core::Device::Cpu)?; // Example size
        Ok(FFNMemoryLayout {
            matrix,
            pointers: Vec::with_capacity(layer_size),
            bias: vec![0.0; layer_size],
            num_used: 0,
            last_k_active: Vec::new(),
        })
    }

    pub fn add_neuron(&mut self, neuron_idx: usize, weight_row: Vec<f32>, bias: f32) -> Result<()> {
        if self.num_used < self.matrix.dim(0)? {
            self.pointers.push(neuron_idx);
            self.bias[self.num_used] = bias;
            self.matrix
                .narrow(0, self.num_used, 1)?
                .copy_from(&Tensor::from_vec(weight_row, (1, 128), &self.matrix.device())?)?;
            self.num_used += 1;
            self.last_k_active.push(neuron_idx);
            Ok(())
        } else {
            Err(candle_core::Error::Msg("Memory layout full".to_string()))
        }
    }
}