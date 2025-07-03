use candle_core::{Tensor, Device, Result, DType};
use std::sync::{Arc, Mutex};

pub struct KVCache {
    device: Device,
    keys: Vec<Tensor>,    // Key activations per layer
    values: Vec<Tensor>,  // Value activations per layer
    n_layers: usize,
    n_heads: usize,
    d_head: usize,
    batch_size: usize,
    seq_length: usize,
    element_size: usize, // Bytes per element
}

impl KVCache {
    pub fn new(device: Device, n_layers: usize, n_heads: usize, d_head: usize, batch_size: usize, seq_length: usize) -> Result<Self> {
        let element_size = std::mem::size_of::<f32>(); // e = 4 bytes for FP32
        let keys = vec![Tensor::zeros((batch_size, seq_length, n_heads, d_head), DType::F32, &device)?; n_layers];
        let values = vec![Tensor::zeros((batch_size, seq_length, n_heads, d_head), DType::F32, &device)?; n_layers];
        Ok(KVCache {
            device,
            keys,
            values,
            n_layers,
            n_heads,
            d_head,
            batch_size,
            seq_length,
            element_size,
        })
    }

    pub fn size(&self) -> usize {
        2 * self.n_layers * self.n_heads * self.d_head * self.element_size * self.batch_size * self.seq_length
    }

    pub fn update(&mut self, layer: usize, keys: Tensor, values: Tensor) -> Result<()> {
        self.keys[layer] = keys;
        self.values[layer] = values;
        Ok(())
    }

    pub fn load(&self, layer: usize, seq_idx: usize) -> Result<(Tensor, Tensor)> {
        let keys = self.keys[layer].narrow(1, seq_idx, 1)?;
        let values = self.values[layer].narrow(1, seq_idx, 1)?;
        Ok((keys, values))
    }
}