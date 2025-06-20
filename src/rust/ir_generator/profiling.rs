use std::time::Instant;
use std::collections::HashMap;


//// Profiler for measuring execution time of different layers in a neural network

pub struct Profiler {
    layer_times: std::collections::HashMap<String, f64>,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            layer_times: HashMap::new(),
        }
    }

    pub fn profile_layer<F>(&mut self, layer_name: &str, operation: F) -> f64
    where
        F: FnOnce() -> (),
    {
        let start = Instant::now();
        operation();
        let duration = start.elapsed().as_secs_f64();
        self.layer_times.insert(layer_name.to_string(), duration);
        duration
    }

    pub fn get_profile(&self) -> Vec<(String, f64)> {
        self.layer_times.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }
}

/// Dynamic quantization for neural network layers

pub struct DynamicQuantizer;

impl DynamicQuantizer {
    pub fn quantize_layer(layer_data: &[f32], bits: u8) -> Vec<u8> {
        let scale = 2.0f32.powi(bits as i32 - 1);
        layer_data.iter().map(|&x| {
            let quantized = (x * scale).round().clamp(0.0, scale - 1.0);
            quantized as u8
        }).collect()
    }

    pub fn dequantize_layer(quantized_data: &[u8], bits: u8) -> Vec<f32> {
        let scale = 2.0f32.powi(bits as i32 - 1);
        quantized_data.iter().map(|&x| x as f32 / scale).collect()
    }
}

