// SPDX-License-Identifier: Apache-2.0
// This file is part of the Zeta Reticula - Fandango  project, which is licensed under the Apache License 2.0.

// This file contains functions to calculate entropy and measure locality of attention data.
// The entropy function calculates the uncertainty in the data distribution,
// while the locality function measures how close values are to the mean of the data.

use actix_web::{web, App, HttpServer, Responder, HttpResponse, HttpRequest, Error as ActixError};
use actix_web::middleware::Logger;
use actix_files::{Files, NamedFile};
use actix_web_actors::ws;
use actix::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Instant, Duration};
use std::env;
use log::{info, error, debug};
use simple_logger::SimpleLogger;
use serde::{Serialize, Deserialize};
use serde_json::json;
use futures_util::stream::StreamExt;
use std::collections::HashMap;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, Linear, Module};
use anyhow::Result as AnyResult;
use std::fs;

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
    fn quantize(weights: &Tensor, bits: usize) -> AnyResult<Self> {
        let device = weights.device();
        let (min_vals, max_vals) = (weights.min()?, weights.max()?);
        let scale = (max_vals.to_scalar::<f32>()? - min_vals.to_scalar::<f32>()?) / ((1 << bits) as f32 - 1.0);
        let zero_point = min_vals.to_scalar::<f32>()? / scale;
        
        let quantized = ((weights / scale)? + zero_point)?.to_dtype(DType::U8)?;
        
        Ok(Self {
            weights: quantized,
            scale,
            zero_point,
            dtype: DType::U8,
        })
    }
    
    fn dequantize(&self) -> AnyResult<Tensor> {
        let dequantized = (self.weights.to_dtype(DType::F32)? * self.scale)? - self.zero_point;
        Ok(dequantized)
    }
    
    fn quantized_matmul(&self, x: &Tensor) -> AnyResult<Tensor> {
        // Simple matmul with dequantization
        let dequantized = self.dequantize()?;
        x.matmul(&dequantized)
    }
}

// WebSocket handler
async fn websocket(req: HttpRequest, stream: web::Payload) -> Result<HttpResponse, ActixError> {
    let (response, session, mut msg_stream) = actix_ws::handle(&req, stream)?;
    
    // Spawn a task to handle the WebSocket connection
    actix_rt::spawn(async move {
        // Send welcome message
        if let Err(e) = session.text("{\"type\":\"system\",\"message\":\"Connected to Fandango WebSocket\"}").await {
            error!("Error sending welcome message: {}", e);
            return;
        }
        
        // Process incoming messages
        while let Some(Ok(msg)) = msg_stream.next().await {
            match msg {
                actix_ws::Message::Ping(bytes) => {
                    if let Err(e) = session.pong(&bytes).await {
                        error!("Error sending pong: {}", e);
                        break;
                    }
                }
                actix_ws::Message::Text(text) => {
                    info!("Received message: {}", text);
                    // Echo the message back
                    if let Err(e) = session.text(text).await {
                        error!("Error sending echo: {}", e);
                        break;
                    }
                }
                actix_ws::Message::Close(reason) => {
                    info!("WebSocket closing: {:?}", reason);
                    break;
                }
                _ => {
                    // Ignore other message types
                    continue;
                }
            }
        }
        
        info!("WebSocket connection closed");
    });
    
    Ok(response)
}

async fn serve_static(path: web::Path<String>) -> Result<NamedFile, std::io::Error> {
    let path = path.into_inner();
    let static_path = Path::new("static").join(&path);
    
    // Default to index.html if the path is a directory or doesn't exist
    if static_path.is_dir() || !static_path.exists() {
        return NamedFile::open("static/index.html");
    }
    
    NamedFile::open(static_path)
}

#[derive(Serialize, Deserialize)]
struct HealthCheckResponse {
    status: String,
    version: String,
    timestamp: String,
    uptime: u64,
}

async fn health_check() -> impl Responder {
    let start_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    let response = HealthCheckResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION", "unknown").to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        uptime: start_time,
    };
    
    HttpResponse::Ok().json(response)
}

async fn handle_index() -> Result<NamedFile, std::io::Error> {
    // Try to serve the static index.html, fall back to a simple response
    match NamedFile::open("static/index.html") {
        Ok(file) => Ok(file),
        Err(_) => {
            let html = r#"
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Fandango</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            text-align: center; 
                            margin-top: 100px;
                            background-color: #f5f5f5;
                        }
                        .container { 
                            max-width: 800px; 
                            margin: 0 auto; 
                            padding: 20px;
                            background: white;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        }
                        h1 { color: #2c3e50; }
                        .status { 
                            color: #27ae60;
                            font-weight: bold;
                            margin: 20px 0;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Fandango Server</h1>
                        <div class="status">Status: Running</div>
                        <p>Welcome to Fandango, an advanced platform for optimizing LLM inference.</p>
                        <p>If you're seeing this message, the web UI is not built. Please run:</p>
                        <pre>cd web-ui && trunk build --release</pre>
                        <p>Then restart the server.</p>
                    </div>
                </body>
                </html>
            "#;
            
            // Create a temporary file with the HTML content
            let temp_dir = std::env::temp_dir();
            let temp_file = temp_dir.join("fandango_index.html");
            std::fs::write(&temp_file, html)?;
            
            NamedFile::open(temp_file)
        }
    }
}

#[actix_web::main]
// Quantization request payload
#[derive(Debug, Deserialize)]
struct QuantizeRequest {
    model_path: String,
    model_name: String,
    bits: usize,
}

// Response for quantization
#[derive(Serialize)]
struct QuantizeResponse {
    status: String,
    model_name: String,
    original_size: usize,
    quantized_size: usize,
    compression_ratio: f32,
}

// Load and quantize a model
async fn quantize_model(
    data: web::Data<AppState>,
    payload: web::Json<QuantizeRequest>,
) -> HttpResponse {
    let model_path = &payload.model_path;
    let model_name = &payload.model_name;
    let bits = payload.bits;
    
    info!("Received quantization request for model: {}", model_path);
    
    // In a real implementation, you would load the model here
    // For demonstration, we'll create a dummy tensor
    let device = Device::Cpu;
    let weights = Tensor::randn(0f32, 1.0, (1024, 1024), &device).unwrap();
    
    match QuantizedModel::quantize(&weights, bits) {
        Ok(quantized) => {
            let original_size = weights.nbytes();
            let quantized_size = quantized.weights.nbytes();
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
            HttpResponse::InternalServerError().json(json!({
                "status": "error",
                "message": format!("Quantization failed: {}", e)
            }))
        }
    }
}

// Perform inference with a quantized model
async fn infer(
    data: web::Data<AppState>,
    model_name: web::Path<String>,
    input: web::Json<Vec<f32>>,
) -> HttpResponse {
    let models = data.models.lock().unwrap();
    
    match models.get(&model_name.into_inner()) {
        Some(model) => {
            let device = Device::Cpu;
            match Tensor::new(input.as_slice(), &device) {
                Ok(input_tensor) => {
                    match model.quantized_matmul(&input_tensor) {
                        Ok(output) => {
                            match output.to_vec1::<f32>() {
                                Ok(result) => HttpResponse::Ok().json(json!({
                                    "status": "success",
                                    "result": result
                                })),
                                Err(e) => HttpResponse::InternalServerError().json(json!({
                                    "status": "error",
                                    "message": format!("Failed to convert output: {}", e)
                                })),
                            }
                        }
                        Err(e) => HttpResponse::InternalServerError().json(json!({
                            "status": "error",
                            "message": format!("Inference failed: {}", e)
                        })),
                    }
                }
                Err(e) => HttpResponse::BadRequest().json(json!({
                    "status": "error",
                    "message": format!("Invalid input: {}", e)
                })),
            }
        }
        None => HttpResponse::NotFound().json(json!({
            "status": "error",
            "message": "Model not found"
        })),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()
        .expect("Failed to initialize logger");

    info!("Starting Fandango server...");

    // Get the port from environment variable or use default 8080
    let port = env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);

    info!("Server running on http://localhost:{}", port);

    // Initialize app state
    let app_state = web::Data::new(AppState {
        models: Mutex::new(HashMap::new()),
    });

    // Start the HTTP server
    let server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .wrap(Logger::default())
            // WebSocket endpoint
            .service(web::resource("/ws").to(websocket))
            // Health check endpoint
            .service(web::resource("/health").to(health_check))
            // Model quantization endpoint
            .service(web::resource("/api/quantize")
                .route(web::post().to(quantize_model)))
            // Model inference endpoint
            .service(web::resource("/api/infer/{model_name}")
                .route(web::post().to(infer)))
            // Static file serving
            .service(web::resource("/{filename:.*}")
                .to(serve_static))
            .default_service(web::get().to(handle_index))
    })
    .bind(("0.0.0.0", port))?
    .run();

    info!("Server started at http://localhost:{}", port);
    info!("WebSocket available at ws://localhost:{}/ws", port);
    
    server.await
}
