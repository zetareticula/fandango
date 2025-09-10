// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.


use actix_web::{web, App, HttpServer, Responder, HttpResponse, post, get};
use actix_web::middleware::Logger;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use log::{info, error};
use simple_logger::SimpleLogger;
use anyhow::Result;

// App state to hold loaded models
struct AppState {
    models: Mutex<HashMap<String, QuantizedModel>>,
}

// Simple quantized model representation
struct QuantizedModel {
    weights: Tensor,
    scale: f32,
    zero_point: f32,
    dtype: DType,
}

impl QuantizedModel {
    fn quantize(weights: &Tensor, bits: usize) -> Result<Self> {
        let device = weights.device();
        let min_val = weights.flatten_all()?.min(0)?;
        let max_val = weights.flatten_all()?.max(0)?;
        let scale = (max_val.to_scalar::<f32>()? - min_val.to_scalar::<f32>()?) / ((1 << bits) as f32 - 1.0);
        let zero_point = -min_val.to_scalar::<f32>()? / scale;
        
        let scale_tensor = Tensor::new(&[scale], weights.device())?;
        let zero_point_tensor = Tensor::new(&[zero_point], weights.device())?;
        let quantized = weights.broadcast_div(&scale_tensor)?.broadcast_add(&zero_point_tensor)?.to_dtype(DType::U8)?;
        
        Ok(Self {
            weights: quantized,
            scale,
            zero_point,
            dtype: DType::U8,
        })
    }
    
    fn dequantize(&self) -> Result<Tensor> {
        let scale_tensor = Tensor::new(&[self.scale], self.weights.device())?;
        let zero_point_tensor = Tensor::new(&[self.zero_point], self.weights.device())?;
        let dequantized = self.weights.to_dtype(DType::F32)?.broadcast_mul(&scale_tensor)?.broadcast_sub(&zero_point_tensor)?;
        Ok(dequantized)
    }
    
    fn quantized_matmul(&self, x: &Tensor) -> Result<Tensor> {
        let dequantized = self.dequantize()?;
        Ok(x.matmul(&dequantized)?)
    }
}

// Request/response types
#[derive(Debug, Deserialize)]
struct QuantizeRequest {
    model_path: String,
    model_name: String,
    bits: usize,
    #[serde(default = "default_dims")]
    dims: Vec<usize>,
}

fn default_dims() -> Vec<usize> {
    vec![1024, 1024]  // Default to 1024x1024 tensor
}

#[derive(Serialize)]
struct QuantizeResponse {
    status: String,
    model_name: String,
    original_size: usize,
    quantized_size: usize,
    compression_ratio: f32,
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    input: Vec<f32>,
}

#[derive(Serialize)]
struct InferenceResponse {
    status: String,
    result: Option<Vec<f32>>,
    error: Option<String>,
}

// API Handlers
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({ "status": "ok" }))
}

async fn quantize_model(
    data: web::Data<AppState>,
    payload: web::Json<QuantizeRequest>,
) -> HttpResponse {
    let model_path = &payload.model_path;
    let model_name = &payload.model_name;
    let bits = payload.bits;
    let dims = &payload.dims;
    
    info!("Received quantization request for model: {}", model_path);
    
    // In a real implementation, load the actual model weights here
    // For demo, create a random tensor of specified dimensions
    let device = Device::Cpu;
    let weights = match Tensor::randn(0f32, 1.0, dims.as_slice(), &device) {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create tensor: {}", e);
            return HttpResponse::BadRequest().json(InferenceResponse {
                status: "error".to_string(),
                result: None,
                error: Some(format!("Invalid tensor dimensions: {}", e)),
            });
        }
    };
    
    match QuantizedModel::quantize(&weights, bits) {
        Ok(quantized) => {
            let original_size = weights.elem_count() * std::mem::size_of::<f32>();
            let quantized_size = quantized.weights.elem_count() * std::mem::size_of::<u8>();
            let compression_ratio = original_size as f32 / quantized_size as f32;
            
            // Store the model
            let mut models = data.models.lock().unwrap();
            models.insert(model_name.clone(), quantized);
            
            HttpResponse::Ok().json(QuantizeResponse {
                status: "success".to_string(),
                model_name: model_name.clone(),
                original_size,
                quantized_size,
                compression_ratio,
            })
        }
        Err(e) => {
            error!("Quantization failed: {}", e);
            HttpResponse::InternalServerError().json(InferenceResponse {
                status: "error".to_string(),
                result: None,
                error: Some(format!("Quantization failed: {}", e)),
            })
        }
    }
}

async fn infer(
    data: web::Data<AppState>,
    model_name: web::Path<String>,
    input: web::Json<InferenceRequest>,
) -> HttpResponse {
    let models = data.models.lock().unwrap();
    
    match models.get(&model_name.into_inner()) {
        Some(model) => {
            let device = Device::Cpu;
            match Tensor::new(input.input.as_slice(), &device) {
                Ok(input_tensor) => match model.quantized_matmul(&input_tensor) {
                    Ok(output) => match output.to_vec1::<f32>() {
                        Ok(result) => HttpResponse::Ok().json(InferenceResponse {
                            status: "success".to_string(),
                            result: Some(result),
                            error: None,
                        }),
                        Err(e) => HttpResponse::InternalServerError().json(InferenceResponse {
                            status: "error".to_string(),
                            result: None,
                            error: Some(format!("Failed to convert output: {}", e)),
                        }),
                    },
                    Err(e) => HttpResponse::InternalServerError().json(InferenceResponse {
                        status: "error".to_string(),
                        result: None,
                        error: Some(format!("Inference failed: {}", e)),
                    }),
                },
                Err(e) => HttpResponse::BadRequest().json(InferenceResponse {
                    status: "error".to_string(),
                    result: None,
                    error: Some(format!("Invalid input: {}", e)),
                }),
            }
        }
        None => HttpResponse::NotFound().json(InferenceResponse {
            status: "error".to_string(),
            result: None,
            error: Some("Model not found".to_string()),
        }),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .unwrap();

    // Initialize app state
    let app_state = web::Data::new(AppState {
        models: Mutex::new(HashMap::new()),
    });

    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);

    info!("Starting Fandango Quantization Server on port {}", port);

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            // Health check endpoint
            .route("/health", web::get().to(health_check))
            // Model quantization endpoint
            .route("/api/quantize", web::post().to(quantize_model))
            // Model inference endpoint
            .route("/api/infer/{model_name}", web::post().to(infer))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
