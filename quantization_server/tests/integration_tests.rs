use std::sync::Once;
use std::time::Duration;
use reqwest::blocking::Client;
use serde_json::json;
use serial_test::serial;

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        // Start the test server in the background
        std::thread::spawn(|| {
            let _ = std::process::Command::new("cargo")
                .args(["run", "--bin", "quantization_server"])
                .env("RUST_LOG", "debug")
                .status();
        });
        
        // Give the server time to start
        std::thread::sleep(Duration::from_secs(2));
    });
}

#[test]
#[serial]
fn test_health_check() {
    setup();
    let client = Client::new();
    
    let response = client.get("http://localhost:8080/health")
        .send()
        .expect("Failed to send request");
    
    assert_eq!(response.status(), 200);
    
    let json: serde_json::Value = response.json().expect("Failed to parse JSON");
    assert_eq!(json["status"], "ok");
}

#[test]
#[serial]
fn test_quantize_endpoint() {
    setup();
    let client = Client::new();
    
    let request = json!({
        "model_path": "test/models/small_model.bin",
        "model_name": "test-model-1",
        "bits": 8,
        "dims": [128, 128]
    });
    
    let response = client.post("http://localhost:8080/api/v1/quantize")
        .json(&request)
        .send()
        .expect("Failed to send request");
    
    assert_eq!(response.status(), 200);
    
    let json: serde_json::Value = response.json().expect("Failed to parse JSON");
    assert_eq!(json["status"], "success");
    assert_eq!(json["model_name"], "test-model-1");
    assert!(json["compression_ratio"].as_f64().unwrap() > 1.0);
}

#[test]
#[serial]
fn test_inference_endpoint() {
    setup();
    let client = Client::new();
    
    // First quantize a model
    let quantize_request = json!({
        "model_path": "test/models/small_model.bin",
        "model_name": "test-inference-model",
        "bits": 8,
        "dims": [128, 128]
    });
    
    let _ = client.post("http://localhost:8080/api/v1/quantize")
        .json(&quantize_request)
        .send()
        .expect("Failed to quantize model");
    
    // Then run inference
    let input = vec![0.1; 128];
    let inference_request = json!({
        "input": input,
        "temperature": 0.7,
        "max_tokens": 10
    });
    
    let response = client.post("http://localhost:8080/api/v1/infer/test-inference-model")
        .json(&inference_request)
        .send()
        .expect("Failed to send inference request");
    
    assert_eq!(response.status(), 200);
    
    let json: serde_json::Value = response.json().expect("Failed to parse JSON");
    assert_eq!(json["status"], "success");
    assert!(json["output"].is_array());
}

#[test]
#[serial]
fn test_error_handling() {
    setup();
    let client = Client::new();
    
    // Test with invalid model name
    let response = client.get("http://localhost:8080/api/v1/models/nonexistent")
        .send()
        .expect("Failed to send request");
    
    assert_eq!(response.status(), 404);
    
    let json: serde_json::Value = response.json().expect("Failed to parse JSON");
    assert_eq!(json["status"], "error");
    assert_eq!(json["error"]["code"], "MODEL_NOT_FOUND");
}
