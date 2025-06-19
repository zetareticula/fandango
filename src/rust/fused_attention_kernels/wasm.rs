use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn init_wasm_attention() -> *mut FusedAttention {
    let device = Device::Cpu; // WASM runs on CPU
    let attention = FusedAttention::new(device).unwrap();
    Box::into_raw(Box::new(attention))
}

#[wasm_bindgen]
pub fn apply_wasm_attention(ptr: *mut FusedAttention, query: Vec<f32>) -> Vec<f32> {
    let attention = unsafe { &mut *ptr };
    let tensor = Tensor::from_vec(query, (query.len(), 1), &attention.device).unwrap();
    let output = attention.apply(&tensor, &tensor, &tensor).unwrap().to_vec1::<f32>().unwrap();
    output
}